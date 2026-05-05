from __future__ import annotations

from types import SimpleNamespace

from ai.repo_meta.category import answer_repo_content_category_summary_question
from ai.structured_skill_summary import build_structured_skill_summary_materials
from app.chat_retrieval_flow import build_topic_summarizer
from app.dialog.state_machine import ConversationState
from app.chat_loop_handlers.guards import _normalize_for_guard


def _looks_like_file_result_set_topic_summary_followup(question: str) -> bool:
    q = _normalize_for_guard(question)
    if not q or len(q) > 16:
        return False
    patterns = (
        "是关于什么的",
        "是讲什么的",
        "是在说什么的",
        "是什么内容",
        "是什么主题",
        "主要讲什么",
        "主要是什么",
    )
    return any(pattern == q for pattern in patterns)


def _build_repo_state_subset_by_paths(repo_state, selected_paths: list[str]):
    selected = {str(path or "").strip() for path in (selected_paths or []) if str(path or "").strip()}
    if not selected:
        return None

    all_paths = [str(path or "").strip() for path in list(getattr(repo_state, "paths", []) or [])]
    subset_paths = [path for path in all_paths if path in selected]
    if not subset_paths:
        return None

    all_files = [
        str(path or "").strip()
        for path in list(getattr(repo_state, "all_files", []) or [])
        if str(path or "").strip() in selected
    ]
    original_times = list(getattr(repo_state, "file_times", []) or [])
    time_by_path = {
        path: original_times[idx]
        for idx, path in enumerate(all_paths)
        if idx < len(original_times)
    }
    subset_file_times = [time_by_path[path] for path in subset_paths if path in time_by_path]
    original_docs = list(getattr(repo_state, "docs", []) or [])
    doc_by_path = {
        path: original_docs[idx]
        for idx, path in enumerate(all_paths)
        if idx < len(original_docs)
    }
    subset_docs = [doc_by_path[path] for path in subset_paths if path in doc_by_path]
    subset_records = [
        dict(record)
        for record in list(getattr(repo_state, "doc_records", []) or [])
        if str(record.get("path", "") or "").strip() in selected
    ]

    return SimpleNamespace(
        paths=subset_paths,
        docs=subset_docs,
        all_files=all_files or list(subset_paths),
        file_times=subset_file_times,
        doc_records=subset_records,
    )


def _try_answer_file_result_set_topic_summary(
    *,
    question: str,
    event_name: str,
    repo_state,
    conversation_state: ConversationState | None,
    model_emb,
    logger,
    ollama_api_url: str,
    ollama_model: str,
) -> str | None:
    if event_name != "result_set_followup":
        return None
    if conversation_state is None:
        return None
    if conversation_state.last_result_set_entity_type != "文件":
        return None
    if conversation_state.last_answer_type != "enumeration_file":
        return None
    if not _looks_like_file_result_set_topic_summary_followup(question):
        return None

    subset_state = _build_repo_state_subset_by_paths(
        repo_state,
        list(conversation_state.last_result_set_items or []),
    )
    if subset_state is None:
        return None

    answer = answer_repo_content_category_summary_question(
        subset_state,
        topic_summarizer=build_topic_summarizer(logger, ollama_api_url, ollama_model),
    )
    if answer:
        logger.info("🛟 [结果集概括] 基于上一轮文件集合做本地主题概括")
        return answer
    return None


def _try_answer_structured_skill_summary(
    *,
    event_name: str,
    repo_state,
    conversation_state: ConversationState | None,
    logger,
) -> str | None:
    if event_name != "structured_skill_summary":
        return None
    if conversation_state is None:
        return None
    if conversation_state.last_result_set_entity_type != "文件":
        return None

    subset_state = _build_repo_state_subset_by_paths(
        repo_state,
        list(conversation_state.last_result_set_items or []),
    )
    if subset_state is None:
        return "当前没有可继承的文件结果集，建议先列出文件或明确指定范围后再归纳。"

    answer = build_structured_skill_summary_materials(subset_state)
    if answer:
        logger.info("🛟 [结构化能力归纳] 基于上一轮文件集合做本地统计归纳")
        return answer
    return None
