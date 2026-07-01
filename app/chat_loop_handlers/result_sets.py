from __future__ import annotations

import re
from types import SimpleNamespace

from ai.repo_meta.category import (
    answer_repo_content_category_summary_question,
    answer_repo_content_type_theme_summary_question,
)
from ai.structured_skill_summary import build_structured_skill_summary_materials
from app.chat_retrieval_flow import build_topic_summarizer
from app.dialog.state_machine import ConversationState
from app.dialog_utils import is_summary_followup_request


def _looks_like_file_result_set_topic_summary_followup(question: str) -> bool:
    return is_summary_followup_request(question)


def _normalize_progressive_summary_output(raw_text: str) -> str:
    lines = [str(line or "").strip() for line in str(raw_text or "").splitlines() if str(line or "").strip()]
    if not lines:
        return ""

    cleaned: list[str] = []
    for line in lines:
        text = re.sub(r"^(?:[-*•]|\d+[.、])\s*", "", line).strip()
        text = re.sub(r"^(?:概括|总结|归纳|总体|整体|主线)\s*[:：]\s*", "", text).strip()
        if text:
            cleaned.append(text)
    if not cleaned:
        return ""

    sentence = "，".join(cleaned)
    sentence = re.sub(r"\s+", "", sentence).strip("，。；; ")
    return f"{sentence}。" if sentence else ""


def _summary_content_length(text: str) -> int:
    return len(re.sub(r"[\s，。！？、,.!?；;：:\-*•\d]+", "", str(text or "")))


def _fallback_compress_previous_summary(previous_summary: str, target_limit: int) -> str:
    labels: list[str] = []
    for line in str(previous_summary or "").splitlines():
        match = re.match(r"^\s*[-*•]\s*(.+?)\s*$", line)
        if match:
            label = match.group(1).strip("，。；; ")
            if label and label not in labels:
                labels.append(label)

    if labels:
        selected: list[str] = []
        for label in labels:
            candidate = "、".join(selected + [label])
            if selected and len(candidate) > target_limit:
                break
            selected.append(label)
            if len(selected) >= 3:
                break
        if selected:
            return f"整体围绕{'、'.join(selected)}。"

    compact = _normalize_progressive_summary_output(previous_summary)
    compact = re.sub(r"^(?:整体看|总体看|这些文档|这些资料|文档集合)", "", compact).strip("，。；; ")
    first_clause = re.split(r"[，；;。]", compact, maxsplit=1)[0].strip()
    if first_clause:
        shortened = first_clause[:target_limit].rstrip("，。；; ")
        return f"{shortened}。" if shortened else ""
    return ""


def _summarize_previous_result_set_summary(
    previous_summary: str,
    *,
    next_level: int,
    topic_summarizer,
) -> str | None:
    previous = str(previous_summary or "").strip()
    if not previous or topic_summarizer is None:
        return None

    previous_length = _summary_content_length(previous)
    level_cap = 42 if next_level <= 2 else 24 if next_level == 3 else 16
    target_limit = max(8, min(level_cap, previous_length - 4))
    prompt = (
        f"下面是上一轮第 {max(1, next_level - 1)} 层概括。请只基于这段概括继续向上抽象，"
        f"压缩成第 {next_level} 层概括。\n"
        "必须遵守：\n"
        "1. 只输出一句完整中文，不要列表、标题或解释\n"
        "2. 保留共同主线，删除并列细项，不得重新展开\n"
        f"3. 正文不超过 {target_limit} 个汉字，并且必须比上一轮更短\n\n"
        f"上一轮概括：\n{previous}"
    )

    for _ in range(2):
        try:
            candidate = _normalize_progressive_summary_output(topic_summarizer(prompt))
        except Exception:
            candidate = ""
        candidate_length = _summary_content_length(candidate)
        if candidate and 0 < candidate_length < previous_length and candidate_length <= target_limit:
            return candidate
        prompt += "\n\n上次输出未满足长度或格式要求，请进一步压缩，只返回一句更短的概括。"

    fallback = _fallback_compress_previous_summary(previous, target_limit)
    if fallback and _summary_content_length(fallback) < previous_length:
        return fallback
    return None


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
    if event_name not in {"result_set_followup", "content_followup", "action_request"}:
        return None
    if conversation_state is None:
        return None
    if conversation_state.last_result_set_entity_type != "文件":
        return None
    if not _looks_like_file_result_set_topic_summary_followup(question):
        return None

    subset_state = _build_repo_state_subset_by_paths(
        repo_state,
        list(conversation_state.last_result_set_items or []),
    )
    if subset_state is None:
        return None

    topic_summarizer = build_topic_summarizer(logger, ollama_api_url, ollama_model)
    previous_summary = str(conversation_state.last_result_set_summary_text or "").strip()
    current_level = max(0, int(conversation_state.last_result_set_summary_level or 0))
    if previous_summary:
        answer = _summarize_previous_result_set_summary(
            previous_summary,
            next_level=current_level + 1,
            topic_summarizer=topic_summarizer,
        )
    else:
        answer = answer_repo_content_type_theme_summary_question(
            subset_state,
            topic_summarizer=topic_summarizer,
        )
        if not answer:
            answer = answer_repo_content_category_summary_question(
                subset_state,
                topic_summarizer=topic_summarizer,
            )
    if answer:
        if previous_summary:
            logger.info(f"🛟 [递进概括] 基于上一轮概括继续压缩至第 {current_level + 1} 层")
        else:
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
