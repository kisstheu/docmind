from __future__ import annotations

import os
from pathlib import Path
import re

from google.genai import types
import requests

from ai.capabilities import (
    answer_repo_meta_question,
    answer_smalltalk,
    answer_system_capability_question,
)
from app.dialog_state_machine import (
    ConversationState,
    apply_event_to_state,
    detect_dialog_event,
)
from app.dialog_state_machine import extract_result_set_from_answer
from retrieval.search_engine import determine_query_flags

from app.chat_retrieval_flow import (
    build_retrieval_materials,
    build_safe_final_prompt,
    build_search_query,
    build_topic_summarizer,
    resolve_route,
)
from app.chat_state_helpers import (
    append_memory,
    print_answer,
    update_state_after_local_answer,
    update_state_after_retrieval_answer,
)
from app.chat_text_utils import (
    maybe_build_direct_lookup_answer,
    maybe_build_file_location_answer,
    maybe_build_related_records_answer,
    normalize_colloquial_question,
)
from app.context_anchor import is_context_dependent_question
from app.dialog_utils import is_followup_question
from app.file_actions.loop import handle_file_action_turn
from infra.file_change_store import FileChangeStore


conversation_state = ConversationState()

CONTEXTLESS_FOLLOWUP_REPLY = (
    "这个问题缺少明确主语或上下文，我先不调用远程模型。"
    "请补充具体对象后再问，例如：请再概括一下 4月1日会议纪要。"
)
_CONTEXTLESS_FOLLOWUP_MARKERS = (
    "关于什么",
    "再概括",
    "还能再概括",
    "更概括",
    "再总结",
    "再归纳",
    "一句话概括",
    "一句话总结",
    "展开一下",
    "详细一点",
    "再详细",
    "继续",
    "然后呢",
    "还有呢",
)
_SUBJECT_HINT_TERMS = (
    "文件",
    "文档",
    "资料",
    "记录",
    "笔记",
    "截图",
    "会议",
    "纪要",
    "公司",
    "企业",
    "项目",
    "人物",
    "人名",
    "代码",
    "仓库",
    "目录",
)
_NO_CONTEXT_ANSWER_MARKERS = (
    "没有检索到足够可靠的参考片段",
    "信息不足",
    "没有可检索文档",
    "没有形成可检索片段",
    "先不调用远程模型",
)


def _normalize_for_guard(text: str) -> str:
    q = (text or "").strip().lower()
    q = re.sub(r"[，。！？、,.!?\s]+", "", q)
    return q


def _looks_like_contextless_followup_question(question: str) -> bool:
    q = _normalize_for_guard(question)
    if not q or len(q) > 24:
        return False

    if any(term in q for term in _SUBJECT_HINT_TERMS):
        return False

    if any(marker in q for marker in _CONTEXTLESS_FOLLOWUP_MARKERS):
        return True

    return is_followup_question(question)


def _has_usable_followup_context(state: ConversationState) -> bool:
    last_answer = (state.last_answer_text or state.last_answer_preview or "").strip()
    if last_answer and not any(marker in last_answer for marker in _NO_CONTEXT_ANSWER_MARKERS):
        return True

    last_query = (state.last_effective_search_query or "").strip()
    if last_query and not _looks_like_contextless_followup_question(last_query):
        return True

    last_content_q = (state.last_content_user_question or "").strip()
    if last_content_q and not _looks_like_contextless_followup_question(last_content_q):
        return True

    return False


def _is_simple_retrieval_turn(question: str, event_name: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if event_name in {"entity_lookup_followup", "result_set_followup", "result_set_expansion_followup"}:
        return True
    if event_name in {"content_followup", "action_request"} and len(q) <= 64:
        return True
    if event_name == "unknown" and len(q) <= 28:
        return True
    return False


def try_handle_contextless_followup(question: str, state: ConversationState, event, logger) -> str | None:
    event_route_hint = getattr(event, "route_hint", None)
    if event_route_hint in {"repo_meta", "smalltalk", "system_capability"}:
        return None

    if not _looks_like_contextless_followup_question(question):
        return None

    if not is_context_dependent_question(question, state.last_effective_search_query):
        return None

    if _has_usable_followup_context(state):
        return None

    logger.info("⛔ [本地短路] 命中无主语追问，跳过检索与远程模型调用")
    return CONTEXTLESS_FOLLOWUP_REPLY


def try_handle_system_capability(route: str, question: str) -> str | None:
    if route != "system_capability":
        return None
    return answer_system_capability_question(question)


def try_handle_repo_meta(
    route: str,
    question: str,
    repo_state,
    model_emb,
    logger,
    prev_content_user_question: str | None,
    ollama_api_url: str,
    ollama_model: str,
):
    global conversation_state

    if route != "repo_meta":
        return None, None

    logger.info("📦 命中 repo_meta，准备本地回答")
    local_answer, local_topic = answer_repo_meta_question(
        question,
        repo_state,
        model_emb=model_emb,
        last_user_question=prev_content_user_question,
        last_local_topic=conversation_state.last_local_topic,
        last_local_answer=(conversation_state.last_answer_text or conversation_state.last_answer_preview),
        topic_summarizer=build_topic_summarizer(logger, ollama_api_url, ollama_model),
    )
    logger.info(f"📦 repo_meta 返回值: {repr(local_answer)[:200]} | topic={local_topic}")

    if local_topic == "time" and "份 Word 文档是" in local_answer:  # 针对时间查询的列表
        items, entity_type = extract_result_set_from_answer(local_answer, "文件")
        conversation_state.last_result_set_items = items
        conversation_state.last_result_set_entity_type = entity_type
        conversation_state.last_result_set_query = question  # 可选，记录原查询
        logger.info(f"🗂️ [结果集提取] 捕获 {len(items)} 个项: {items}")

    return local_answer, local_topic


def _call_local_ollama_answer(
    prompt: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
    *,
    timeout_sec: float,
    log_prefix: str,
) -> str | None:
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
    }
    try:
        resp = requests.post(ollama_api_url, json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        text = (resp.json().get("response") or "").strip()
        if not text:
            return None
        compact = " ".join(text.split())
        return compact[:180]
    except Exception as e:
        logger.warning(f"⚠️ {log_prefix} 本地模型回答失败，回退固定文案：{e}")
        return None


def _is_smalltalk_local_llm_enabled() -> bool:
    raw = (os.getenv("DOCMIND_SMALLTALK_LOCAL_LLM") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _build_smalltalk_prompt(
    question: str,
    *,
    prev_user_question: str | None = None,
    prev_answer_preview: str | None = None,
    last_route: str | None = None,
) -> str:
    prev_q = (prev_user_question or "").strip() or "unknown"
    prev_a = (prev_answer_preview or "").strip() or "unknown"
    prev_route = (last_route or "").strip() or "unknown"
    return f"""
你是 DocMind 助手。请对用户的闲聊做自然简短回复。
要求：
1) 直接回答用户，不要说“超出范围”。
2) 回答控制在 1-2 句中文。
3) 不要虚构你看过用户本地文档。
4) 涉及你自身生理状态时，以 AI 助手身份回答。
5) 如果当前问句是残句/接话，先结合上一轮语境补全语义，再直接回答。

上一轮路由：{prev_route}
上一轮用户问题：{prev_q}
上一轮回答摘要：{prev_a}
用户问题：{question}
回答：
""".strip()


def _answer_smalltalk_with_local_llm(
    question: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
    *,
    prev_user_question: str | None = None,
    prev_answer_preview: str | None = None,
    last_route: str | None = None,
) -> str | None:
    if not _is_smalltalk_local_llm_enabled():
        return None
    prompt = _build_smalltalk_prompt(
        question,
        prev_user_question=prev_user_question,
        prev_answer_preview=prev_answer_preview,
        last_route=last_route,
    )
    return _call_local_ollama_answer(
        prompt=prompt,
        ollama_api_url=ollama_api_url,
        ollama_model=ollama_model,
        logger=logger,
        timeout_sec=20,
        log_prefix="smalltalk",
    )


def try_handle_smalltalk(
    route: str,
    question: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
    prefetched_smalltalk_answer: str | None = None,
) -> str | None:
    global conversation_state

    if route != "smalltalk":
        return None
    local = answer_smalltalk(question, dialog_state=conversation_state)
    if local is not None:
        return local

    prefetched = (prefetched_smalltalk_answer or "").strip()
    if prefetched:
        return prefetched

    local_llm_answer = _answer_smalltalk_with_local_llm(
        question=question,
        ollama_api_url=ollama_api_url,
        ollama_model=ollama_model,
        logger=logger,
        prev_user_question=getattr(conversation_state, "last_user_question", None),
        prev_answer_preview=getattr(conversation_state, "last_answer_preview", None),
        last_route=getattr(conversation_state, "last_route", None),
    )
    if local_llm_answer:
        return local_llm_answer

    return "这个我先按闲聊处理，不走文档检索。你也可以继续问文档相关内容。"


def _is_out_of_scope_local_llm_enabled() -> bool:
    raw = (os.getenv("DOCMIND_OUT_OF_SCOPE_LOCAL_LLM") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _build_out_of_scope_prompt(
    question: str,
    *,
    prev_user_question: str | None = None,
    prev_answer_preview: str | None = None,
    last_route: str | None = None,
) -> str:
    prev_q = (prev_user_question or "").strip() or "unknown"
    prev_a = (prev_answer_preview or "").strip() or "unknown"
    prev_route = (last_route or "").strip() or "unknown"
    return f"""
你是 DocMind 助手。用户这次问题超出“文档整理与检索”主线，但你可以做简短对话回答。
要求：
1) 直接回答用户，不要提“超出范围/越界”。
2) 回答控制在 1-2 句中文。
3) 不要声称你看过用户的本地文档或检索结果。
4) 涉及实时外部信息（天气/新闻/股价等）时，如无法确认请明确说明无法获取实时数据。
5) 涉及你自身生理状态时，以 AI 助手身份自然回答。
6) 如果当前问句是纠偏或残句，先结合上一轮语境补全后再回答。

上一轮路由：{prev_route}
上一轮用户问题：{prev_q}
上一轮回答摘要：{prev_a}
用户问题：{question}
回答：
""".strip()


def _answer_out_of_scope_with_local_llm(
    question: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
    *,
    prev_user_question: str | None = None,
    prev_answer_preview: str | None = None,
    last_route: str | None = None,
) -> str | None:
    if not _is_out_of_scope_local_llm_enabled():
        return None

    prompt = _build_out_of_scope_prompt(
        question,
        prev_user_question=prev_user_question,
        prev_answer_preview=prev_answer_preview,
        last_route=last_route,
    )
    return _call_local_ollama_answer(
        prompt=prompt,
        ollama_api_url=ollama_api_url,
        ollama_model=ollama_model,
        logger=logger,
        timeout_sec=20,
        log_prefix="out_of_scope",
    )


def try_handle_out_of_scope(
    route: str,
    question: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
    effective_question: str | None = None,
) -> str | None:
    global conversation_state

    if route != "out_of_scope":
        return None

    merged_question = (effective_question or "").strip()
    question_for_answer = merged_question or question

    local = answer_smalltalk(question, dialog_state=conversation_state)
    if local is not None:
        return local
    if question_for_answer != question:
        local = answer_smalltalk(question_for_answer, dialog_state=conversation_state)
        if local is not None:
            return local

    local_llm_answer = _answer_out_of_scope_with_local_llm(
        question=question_for_answer,
        ollama_api_url=ollama_api_url,
        ollama_model=ollama_model,
        logger=logger,
        prev_user_question=getattr(conversation_state, "last_user_question", None),
        prev_answer_preview=getattr(conversation_state, "last_answer_preview", None),
        last_route=getattr(conversation_state, "last_route", None),
    )
    if local_llm_answer:
        return local_llm_answer

    return "这个问题超出当前 DocMind 的文档整理范围，我先不调用远程模型。你可以继续问文档、记录、项目或人物相关内容。"


def run_chat_loop(
    repo_state,
    model_emb,
    client,
    model_id: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
    *,
    notes_dir: Path,
    change_log_file: Path,
    question_recorder=None,
):
    global conversation_state
    change_store = FileChangeStore(change_log_file)

    chat_config = types.GenerateContentConfig(
        system_instruction=(
            "你是用户的个人笔记助理，擅长根据给定材料回答问题、整理线索、做有限归纳。\n"
            f"当前知识库共有 {len(repo_state.paths)} 个文件。\n"
            f"时间跨度参考：最早文件是 {repo_state.earliest_note}；最新文件是 {repo_state.latest_note}。\n"
            "你只能依据本轮给出的参考片段回答，不要假设自己看过所有文件全文。\n"
            "有证据就答，没有证据就明确说信息不足。\n"
            "不要因为名字相似、简称相似，就擅自把不同人物、公司、项目混为一谈。\n"
        ),
        temperature=0.4,
    )

    memory_buffer: list[str] = []
    current_focus_file = None
    last_relevant_indices = []

    print("=================================")
    print("🤖：你好！我是你的 DocMind 随身助理。你可以问我任何问题。")

    while True:
        raw_question = input("\n问：")
        if raw_question.strip().lower() in ["q", "quit", "exit"]:
            break
        if not raw_question.strip():
            continue

        question = normalize_colloquial_question(raw_question)
        if question_recorder is not None:
            question_recorder.record(raw_question, normalized_question=question)

        try:
            import time
            start_qa = time.time()

            file_action_handled, conversation_state, current_focus_file = handle_file_action_turn(
                question=question,
                start_qa=start_qa,
                state=conversation_state,
                memory_buffer=memory_buffer,
                current_focus_file=current_focus_file,
                repo_state=repo_state,
                notes_dir=notes_dir,
                change_store=change_store,
            )
            if file_action_handled:
                continue

            prev_content_user_question = conversation_state.last_content_user_question

            event = detect_dialog_event(question, conversation_state,logger)
            local_answer = try_handle_contextless_followup(
                question=question,
                state=conversation_state,
                event=event,
                logger=logger,
            )
            if local_answer is not None:
                print_answer(local_answer, start_qa)
                append_memory(memory_buffer, question, local_answer)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=local_answer,
                    route="normal_retrieval",
                    local_topic=None,
                    is_content_answer=False,
                )
                continue

            conversation_state = apply_event_to_state(conversation_state, event)

            route_info = resolve_route(question, event, ollama_api_url, ollama_model, logger, state=conversation_state)
            route = route_info["route"]
            prefetched_smalltalk_answer = route_info.get("smalltalk_reply", "")
            route_question_input = route_info.get("route_question_input", question)

            # 1) system capability
            local_answer = try_handle_system_capability(route, question)
            if local_answer is not None:
                print_answer(local_answer, start_qa)
                append_memory(memory_buffer, question, local_answer)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=local_answer,
                    route="system_capability",
                    local_topic=None,
                    is_content_answer=False,
                )
                continue

            # 2) repo meta
            local_answer, local_topic = try_handle_repo_meta(
                route,
                question,
                repo_state,
                model_emb,
                logger,
                prev_content_user_question,
                ollama_api_url,
                ollama_model,
            )
            if local_answer is not None:
                print_answer(local_answer, start_qa)
                append_memory(memory_buffer, question, local_answer)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=local_answer,
                    route="repo_meta",
                    local_topic=local_topic,
                    is_content_answer=True,
                )
                continue

            # 3) smalltalk
            local_answer = try_handle_smalltalk(
                route=route,
                question=question,
                ollama_api_url=ollama_api_url,
                ollama_model=ollama_model,
                logger=logger,
                prefetched_smalltalk_answer=prefetched_smalltalk_answer,
            )
            if local_answer is not None:
                print_answer(local_answer, start_qa)
                append_memory(memory_buffer, question, local_answer)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=local_answer,
                    route="smalltalk",
                    local_topic=None,
                    is_content_answer=False,
                )
                continue

            # 4) out_of_scope
            local_answer = try_handle_out_of_scope(
                route=route,
                question=question,
                ollama_api_url=ollama_api_url,
                ollama_model=ollama_model,
                logger=logger,
                effective_question=route_question_input,
            )
            if local_answer is not None:
                print_answer(local_answer, start_qa)
                append_memory(memory_buffer, question, local_answer)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=local_answer,
                    route="out_of_scope",
                    local_topic=None,
                    is_content_answer=False,
                )
                continue

            # 5) normal retrieval
            conversation_state.mode = "content"
            conversation_state.last_user_question = question
            conversation_state.last_route = "normal_retrieval"
            conversation_state.last_local_topic = None

            flags = determine_query_flags(question)

            search_query, context_anchor = build_search_query(
                question=question,
                event=event,
                flags=flags,
                memory_buffer=memory_buffer,
                last_effective_search_query=conversation_state.last_effective_search_query,
                last_user_question=conversation_state.last_content_user_question,
                last_answer_type=conversation_state.last_answer_type,
                last_result_set_items=conversation_state.last_result_set_items,
                last_result_set_entity_type=conversation_state.last_result_set_entity_type,
                last_relevant_indices=last_relevant_indices,
                logger=logger,
                ollama_api_url=ollama_api_url,
                ollama_model=ollama_model,
            )

            if not flags["skip_retrieval"] and search_query.strip():
                conversation_state.last_effective_search_query = search_query.strip()

            materials = build_retrieval_materials(
                question=question,
                search_query=search_query,
                context_anchor=context_anchor,
                flags=flags,
                repo_state=repo_state,
                model_emb=model_emb,
                logger=logger,
                current_focus_file=current_focus_file,
                last_relevant_indices=last_relevant_indices,
                event=event,
            )
            current_focus_file = materials["current_focus_file"]
            last_relevant_indices = materials["relevant_indices"]

            related_records_answer = maybe_build_related_records_answer(
                question=question,
                relevant_indices=last_relevant_indices,
                repo_state=repo_state,
            )
            if related_records_answer:
                logger.info("🧩 [相关记录稳态回答] 使用本地列举，避免生成阶段漏列")
                print_answer(related_records_answer, start_qa)
                append_memory(memory_buffer, question, related_records_answer)
                conversation_state = update_state_after_retrieval_answer(
                    conversation_state,
                    question,
                    related_records_answer,
                    logger,
                    event_name=event.name,
                )
                continue

            local_file_locator_answer = maybe_build_file_location_answer(
                question=question,
                search_query=search_query,
                relevant_indices=last_relevant_indices,
                repo_state=repo_state,
                logger=logger,
                allow_followup_inference=(
                    conversation_state.last_answer_type == "enumeration_file"
                    and event.name in {"content_followup", "result_set_followup", "result_set_expansion_followup"}
                ),
            )
            if local_file_locator_answer:
                logger.info("🧩 [文件定位稳态回答] 使用本地规则直接回答，跳过远程模型生成")
                print_answer(local_file_locator_answer, start_qa)
                append_memory(memory_buffer, question, local_file_locator_answer)
                conversation_state = update_state_after_retrieval_answer(
                    conversation_state,
                    question,
                    local_file_locator_answer,
                    logger,
                    event_name=event.name,
                )
                continue

            local_entity_mapping_answer = maybe_build_direct_lookup_answer(
                question=question,
                search_query=search_query,
                relevant_indices=last_relevant_indices,
                repo_state=repo_state,
                logger=logger,
                allow_followup_inference=(
                    (
                        (
                            bool(conversation_state.last_result_set_items)
                            and str(conversation_state.last_answer_type or "").startswith("enumeration_")
                        )
                        or (
                            "可直接核对的证据"
                            in str(conversation_state.last_answer_text or conversation_state.last_answer_preview or "")
                        )
                    )
                    and event.name in {
                        "unknown",
                        "content_followup",
                        "result_set_followup",
                        "result_set_expansion_followup",
                        "entity_lookup_followup",
                    }
                ),
            )
            if local_entity_mapping_answer:
                logger.info("🧩 [直接检索稳态回答] 使用本地证据抽取，跳过远程模型生成")
                print_answer(local_entity_mapping_answer, start_qa)
                append_memory(memory_buffer, question, local_entity_mapping_answer)
                conversation_state = update_state_after_retrieval_answer(
                    conversation_state,
                    question,
                    local_entity_mapping_answer,
                    logger,
                    event_name=event.name,
                )
                continue

            if route == "normal_retrieval" and _is_simple_retrieval_turn(question, event.name):
                forced_local_answer = maybe_build_direct_lookup_answer(
                    question=question,
                    search_query=search_query,
                    relevant_indices=last_relevant_indices,
                    repo_state=repo_state,
                    logger=logger,
                    allow_followup_inference=True,
                    force_local_evidence=True,
                )
                if forced_local_answer:
                    logger.info("🧩 [简单检索本地兜底] 跳过远程模型生成")
                    print_answer(forced_local_answer, start_qa)
                    append_memory(memory_buffer, question, forced_local_answer)
                    conversation_state = update_state_after_retrieval_answer(
                        conversation_state,
                        question,
                        forced_local_answer,
                        logger,
                        event_name=event.name,
                    )
                    continue

            if (
                route == "normal_retrieval"
                and not materials["context_text"].strip()
                and not materials["inventory_candidates_text"].strip()
            ):
                if not list(getattr(repo_state, "paths", []) or []):
                    fallback_answer = "当前知识库还没有可检索文档。请先在笔记目录中放入支持的文件（如 .md/.txt/.pdf/.png 等），再来提问。"
                elif not list(getattr(repo_state, "chunk_texts", []) or []):
                    fallback_answer = "当前文档还没有形成可检索片段，请先检查文件读取/索引是否成功。"
                else:
                    fallback_answer = "这次没有检索到足够可靠的参考片段，所以我先不乱回答。你可以再具体一点。"
                print_answer(fallback_answer, start_qa)
                append_memory(memory_buffer, question, fallback_answer)
                conversation_state = update_state_after_retrieval_answer(
                    conversation_state,
                    question,
                    fallback_answer,
                    logger,
                    event_name=event.name,
                )
                continue

            final_prompt = build_safe_final_prompt(
                memory_buffer=memory_buffer,
                current_focus_file=current_focus_file,
                inventory_candidates_text=materials["inventory_candidates_text"],
                context_text=materials["context_text"],
                timeline_evidence_text=materials["timeline_evidence_text"],
                question=question,
                event_name=event.name,
                result_set_items=conversation_state.last_result_set_items,
            )

            logger.info("🛰️ [远程模型生成] 进入生成阶段，开始调用远程大模型")
            response = client.models.generate_content(
                model=model_id,
                contents=final_prompt,
                config=chat_config,
            )

            answer_text = response.text or "这次我没有生成有效回答。"
            print_answer(answer_text, start_qa)
            append_memory(memory_buffer, question, answer_text)
            conversation_state = update_state_after_retrieval_answer(
                conversation_state,
                question,
                answer_text,
                logger,
                event_name=event.name,
            )

        except Exception as e:
            err = str(e)
            if "UNEXPECTED_EOF_WHILE_READING" in err or "EOF occurred in violation of protocol" in err:
                logger.error( "模型服务连接被中途断开，可能是代理或网络波动导致，请重试。")
            logger.error(f"\n调用失败: {e}")
