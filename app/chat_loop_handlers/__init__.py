from __future__ import annotations

from google.genai import types

from ai.capabilities import (
    answer_repo_meta_question,
    answer_smalltalk,
    answer_system_capability_question,
)
from app.chat_loop_llm import _answer_out_of_scope_with_local_llm, _answer_smalltalk_with_local_llm
from app.chat_retrieval_flow import (
    build_remote_topic_summarizer,
    build_topic_summarizer,
    build_topic_summarizer_with_remote_fallback,
)
from app.chat_text.lookup_answer_main import maybe_build_direct_lookup_answer
from app.context_anchor import is_context_dependent_question
from app.dialog.state_machine import ConversationState, extract_result_set_from_answer
from app.chat_loop_handlers.guards import (
    CONTEXTLESS_FOLLOWUP_REPLY,
    _has_usable_followup_context,
    _looks_like_contextless_followup_question,
    is_simple_retrieval_turn,
    looks_like_analytic_retrieval_question,
)
from app.chat_loop_handlers.result_sets import (
    _looks_like_file_result_set_topic_summary_followup,
    _try_answer_file_result_set_topic_summary,
    _try_answer_structured_skill_summary,
)


def build_chat_config(repo_state):
    return types.GenerateContentConfig(
        system_instruction=(
            "你是一个只基于本地资料回答的助手。"
            f"当前仓库共有 {len(repo_state.paths)} 个文件。"
            f"最早记录：{repo_state.earliest_note}；最新记录：{repo_state.latest_note}。"
            "请优先给出可核对依据；证据不足时明确说明，不要编造。"
        ),
        temperature=0.4,
    )


def try_handle_contextless_followup(
    question: str,
    state: ConversationState,
    event,
    logger,
    *,
    has_focused_document_reference: bool = False,
) -> str | None:
    route_hint = getattr(event, "route_hint", None)
    if route_hint in {"repo_meta", "smalltalk", "system_capability"}:
        return None

    if not _looks_like_contextless_followup_question(question):
        return None
    if not is_context_dependent_question(question, state.last_effective_search_query):
        return None
    if has_focused_document_reference:
        return None
    if _has_usable_followup_context(state):
        return None

    logger.info("🛝 [本地短路] 命中无主语追问，跳过检索和远程模型调用")
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
    conversation_state: ConversationState,
    client=None,
    model_id: str | None = None,
):
    if route != "repo_meta":
        return None, None

    logger.info("📷 命中 repo_meta，准备本地回答")
    remote_topic_summarizer = (
        build_remote_topic_summarizer(logger, client, model_id)
        if client is not None and model_id
        else None
    )
    topic_summarizer = (
        build_topic_summarizer_with_remote_fallback(
            logger,
            ollama_api_url,
            ollama_model,
            client,
            model_id,
        )
        if remote_topic_summarizer is not None
        else build_topic_summarizer(logger, ollama_api_url, ollama_model)
    )
    local_answer, local_topic = answer_repo_meta_question(
        question,
        repo_state,
        model_emb=model_emb,
        last_user_question=prev_content_user_question,
        last_local_topic=conversation_state.last_local_topic,
        last_local_answer=(conversation_state.last_answer_text or conversation_state.last_answer_preview),
        category_context_answer=getattr(conversation_state, "last_category_context_answer", None),
        topic_summarizer=topic_summarizer,
        fallback_topic_summarizer=remote_topic_summarizer,
    )
    logger.info(f"📷 repo_meta 返回值: {repr(local_answer)[:200]} | topic={local_topic}")

    if local_topic == "time" and "Word 文档" in str(local_answer):
        items, entity_type = extract_result_set_from_answer(local_answer, "文件")
        conversation_state.last_result_set_items = items
        conversation_state.last_result_set_entity_type = entity_type
        conversation_state.last_result_set_query = question
        logger.info(f"🔎 [结果集提取] 捕获 {len(items)} 个项: {items}")

    return local_answer, local_topic


def try_handle_smalltalk(
    route: str,
    question: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
    conversation_state: ConversationState,
    prefetched_smalltalk_answer: str | None = None,
) -> str | None:
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


def try_handle_out_of_scope(
    route: str,
    question: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
    conversation_state: ConversationState,
    effective_question: str | None = None,
) -> str | None:
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


def try_handle_retrieval_force_local_or_empty_context(
    *,
    route: str,
    question: str,
    event_name: str,
    search_query: str,
    relevant_indices,
    repo_state,
    materials: dict,
    conversation_state: ConversationState | None = None,
    model_emb=None,
    logger,
    ollama_api_url: str | None = None,
    ollama_model: str | None = None,
    prefer_content_answer: bool = False,
) -> str | None:
    if route == "normal_retrieval":
        structured_skill_summary_answer = _try_answer_structured_skill_summary(
            event_name=event_name,
            repo_state=repo_state,
            conversation_state=conversation_state,
            logger=logger,
        )
        if structured_skill_summary_answer:
            return structured_skill_summary_answer

        result_set_summary_answer = _try_answer_file_result_set_topic_summary(
            question=question,
            event_name=event_name,
            repo_state=repo_state,
            conversation_state=conversation_state,
            model_emb=model_emb,
            logger=logger,
            ollama_api_url=ollama_api_url or "",
            ollama_model=ollama_model or "",
        )
        if result_set_summary_answer:
            return result_set_summary_answer

    if route == "normal_retrieval" and not prefer_content_answer and is_simple_retrieval_turn(question, event_name):
        forced_local_answer = maybe_build_direct_lookup_answer(
            question=question,
            search_query=search_query,
            relevant_indices=relevant_indices,
            repo_state=repo_state,
            logger=logger,
            allow_followup_inference=True,
            force_local_evidence=True,
        )
        if forced_local_answer:
            logger.info("🛟 [本地兜底] 简单检索问题优先给本地证据")
            return forced_local_answer

    if (
        route == "normal_retrieval"
        and not (materials.get("context_text") or "").strip()
        and not (materials.get("inventory_candidates_text") or "").strip()
    ):
        if not list(getattr(repo_state, "paths", []) or []):
            return "仓库里还没有可检索的文件。请先放入 `.md/.txt/.pdf/.png` 等资料后再试。"
        if not list(getattr(repo_state, "chunk_texts", []) or []):
            return "已发现文件，但还没有可检索片段；请确认文件内容可读取。"
        return "本轮没有检索到可用证据，建议换更具体关键词或指定文件名重试。"

    return None
