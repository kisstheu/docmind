from __future__ import annotations

import re

from google.genai import types

from ai.capabilities import (
    answer_repo_meta_question,
    answer_smalltalk,
    answer_system_capability_question,
)
from app.chat_loop_llm import _answer_out_of_scope_with_local_llm, _answer_smalltalk_with_local_llm
from app.chat_retrieval_flow import build_topic_summarizer
from app.chat_text_utils import maybe_build_direct_lookup_answer
from app.context_anchor import is_context_dependent_question
from app.dialog_state_machine import ConversationState, extract_result_set_from_answer
from app.dialog_utils import is_followup_question

CONTEXTLESS_FOLLOWUP_REPLY = (
    "这个问题缺少明确主语或上下文，我先不调用远程模型。"
    "请补充具体对象后再问，例如：请再概括一下 3 月 4 日会议纪要。"
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
    "还有吗",
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

_ANALYTIC_RETRIEVAL_MARKERS = (
    "\u4e3a\u4ec0\u4e48",
    "\u600e\u4e48",
    "\u5982\u4f55",
    "\u5206\u6790",
    "\u603b\u7ed3",
    "\u6bd4\u8f83",
    "\u533a\u522b",
    "\u4e3b\u8981\u96c6\u4e2d",
    "\u96c6\u4e2d\u5728\u54ea",
    "\u54ea\u4e9b\u65b9\u5411",
    "\u6280\u672f\u65b9\u5411",
    "\u65b9\u5411\u4e4b\u95f4",
    "\u4e4b\u95f4\u7684\u5173\u7cfb",
    "\u5173\u7cfb\u548c\u7ec4\u5408",
    "\u7ec4\u5408\u60c5\u51b5",
    "\u5206\u5e03",
    "\u8d8b\u52bf",
    "\u7ed3\u6784",
    "\u6a21\u5f0f",
    "\u5171\u73b0",
    "\u642d\u914d",
    "\u54ea\u4e00\u7c7b",
    "\u54ea\u7c7b",
    "\u9700\u6c42\u6700\u591a",
    "\u6570\u91cf",
    "\u5360\u6bd4",
    "\u6309\u6570\u91cf",
    "\u6309\u5360\u6bd4",
    "\u7b80\u5355\u8bf4\u660e",
    "\u6280\u672f\u6808",
    "\u7ecf\u9a8c\u8981\u6c42",
    "\u5207\u5165\u53e3",
    "\u95e8\u69db",
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


def _normalize_for_guard(text: str) -> str:
    q = (text or "").strip().lower()
    return re.sub(r"[，。！？?.!?\s]+", "", q)


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


def looks_like_analytic_retrieval_question(question: str) -> bool:
    q = _normalize_for_guard(question)
    if not q:
        return False
    if any(word in q for word in ("\u6587\u4ef6", "\u6587\u6863", "\u8bb0\u5f55")) and re.search(
        r"(?:\u54ea\u4e2a|\u54ea\u4efd|\u54ea\u7bc7|\u5728\u54ea)",
        q,
    ):
        return False
    if any(marker in q for marker in _ANALYTIC_RETRIEVAL_MARKERS):
        return True
    if re.search(r"\u8fd9\u4e9b.*\u65b9\u5411", q):
        return True
    if re.search(r"\u54ea\u4e9b.*\u65b9\u5411", q):
        return True
    if re.search(r"\u54ea.*\u7c7b.*\u6700\u591a", q):
        return True
    if re.search(r"(?:\u8981\u6c42|\u6280\u80fd|\u638c\u63e1|\u6280\u672f\u6808|\u80fd\u529b|\u7ecf\u9a8c).*(?:\u54ea\u4e9b|\u4ec0\u4e48|\u76f8\u5bf9\u8f83\u4f4e|\u5207\u5165\u53e3)", q):
        return True
    if re.search(r"\u54ea.*(?:\u6280\u672f|\u6280\u80fd)", q):
        return True
    if re.search(r"\u66f4\u5bb9\u6613.*\u5207\u5165\u53e3", q):
        return True
    if re.search(r".*\u4e4b\u95f4.*\u5173\u7cfb", q):
        return True
    return False


def is_simple_retrieval_turn(question: str, event_name: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if looks_like_analytic_retrieval_question(q):
        return False
    if event_name in {"entity_lookup_followup", "result_set_followup", "result_set_expansion_followup"}:
        return True
    if event_name in {"content_followup", "action_request"} and len(q) <= 64:
        return True
    if event_name == "unknown" and len(q) <= 28:
        return True
    return False


def try_handle_contextless_followup(question: str, state: ConversationState, event, logger) -> str | None:
    route_hint = getattr(event, "route_hint", None)
    if route_hint in {"repo_meta", "smalltalk", "system_capability"}:
        return None

    if not _looks_like_contextless_followup_question(question):
        return None
    if not is_context_dependent_question(question, state.last_effective_search_query):
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
):
    if route != "repo_meta":
        return None, None

    logger.info("📷 命中 repo_meta，准备本地回答")
    local_answer, local_topic = answer_repo_meta_question(
        question,
        repo_state,
        model_emb=model_emb,
        last_user_question=prev_content_user_question,
        last_local_topic=conversation_state.last_local_topic,
        last_local_answer=(conversation_state.last_answer_text or conversation_state.last_answer_preview),
        category_context_answer=getattr(conversation_state, "last_category_context_answer", None),
        topic_summarizer=build_topic_summarizer(logger, ollama_api_url, ollama_model),
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
    logger,
) -> str | None:
    if route == "normal_retrieval" and is_simple_retrieval_turn(question, event_name):
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
