from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Callable, Sequence

from ai.capability_common import extract_fine_topics
from ai.repo_meta.category.assignment import (
    _assign_records_to_summary_labels_with_local_llm,
    build_local_category_assignment_map,
)
from ai.repo_meta.category.shared import (
    _extract_count_target_topic,
    _extract_summary_labels,
    _match_category_label,
    _match_category_label_with_local_llm,
)


def _extract_last_category_focus_label(answer_text: str | None) -> str | None:
    text = str(answer_text or "").strip()
    if not text:
        return None

    match = re.search(r"板块[“\"](?P<label>[^”\"]+)[”\"]", text)
    if not match:
        return None

    label = match.group("label").strip()
    return label or None


def _build_repo_state_subset(repo_state, matched_paths: Sequence[str]):
    selected = {str(path or "").strip() for path in matched_paths if str(path or "").strip()}
    if not selected:
        return None

    subset_paths = [path for path in list(getattr(repo_state, "paths", []) or []) if path in selected]
    subset_all_files = [path for path in list(getattr(repo_state, "all_files", []) or []) if path in selected]

    original_paths = list(getattr(repo_state, "paths", []) or [])
    original_times = list(getattr(repo_state, "file_times", []) or [])
    time_by_path = {
        str(path or "").strip(): original_times[idx]
        for idx, path in enumerate(original_paths)
        if idx < len(original_times)
    }
    subset_file_times = [time_by_path[path] for path in subset_paths if path in time_by_path]

    subset_records = [
        dict(record)
        for record in list(getattr(repo_state, "doc_records", []) or [])
        if str(record.get("path", "") or "").strip() in selected
    ]

    return SimpleNamespace(
        paths=subset_paths,
        all_files=subset_all_files or list(subset_paths),
        file_times=subset_file_times,
        doc_records=subset_records,
    )


def resolve_repo_content_category_scope(
    question: str,
    repo_state,
    previous_summary: str | None,
    last_local_answer: str | None = None,
    model_emb=None,
    topic_summarizer: Callable[[str], str] | None = None,
) -> tuple[str | None, list[str]]:
    category_labels = _extract_summary_labels(previous_summary)
    if not category_labels:
        return None, []

    target_topic = question
    matched_label = _match_category_label(target_topic, category_labels)
    if not matched_label:
        extracted_target = _extract_count_target_topic(question)
        if extracted_target:
            target_topic = extracted_target
            matched_label = _match_category_label(target_topic, category_labels)
    if not matched_label and last_local_answer:
        last_focus_label = _extract_last_category_focus_label(last_local_answer)
        if last_focus_label:
            target_topic = last_focus_label
            matched_label = _match_category_label(last_focus_label, category_labels)
    if not matched_label:
        matched_label = _match_category_label_with_local_llm(
            target_topic=target_topic,
            category_labels=category_labels,
            topic_summarizer=topic_summarizer,
        )
    if not matched_label:
        return None, []

    _, path_map = build_local_category_assignment_map(
        repo_state,
        previous_summary=previous_summary,
        model_emb=model_emb,
    )
    if not path_map:
        path_map = _assign_records_to_summary_labels_with_local_llm(
            repo_state,
            category_labels=category_labels,
            topic_summarizer=topic_summarizer,
        )
    if not path_map:
        return matched_label, []

    matched_paths = [
        path
        for path in list(getattr(repo_state, "paths", []) or [])
        if path_map.get(path) == matched_label
    ]
    return matched_label, matched_paths


def answer_repo_content_category_label_drilldown_question(
    question: str,
    repo_state,
    previous_summary: str | None,
    last_local_answer: str | None = None,
    model_emb=None,
    topic_summarizer: Callable[[str], str] | None = None,
) -> str | None:
    category_labels = _extract_summary_labels(previous_summary)
    if not category_labels:
        return None

    matched_label = _match_category_label(question, category_labels)
    if not matched_label and last_local_answer:
        matched_label = _extract_last_category_focus_label(last_local_answer)
        if matched_label:
            matched_label = _match_category_label(matched_label, category_labels)
    if not matched_label:
        return None

    _, path_map = build_local_category_assignment_map(
        repo_state,
        previous_summary=previous_summary,
        model_emb=model_emb,
    )
    if not path_map:
        path_map = _assign_records_to_summary_labels_with_local_llm(
            repo_state,
            category_labels=category_labels,
            topic_summarizer=topic_summarizer,
        )
    if not path_map:
        return None

    matched_paths = [path for path, category in path_map.items() if category == matched_label]
    if not matched_paths:
        return f"按刚才的板块“{matched_label}”继续往下看，当前没有足够稳定的子分类可拆。"

    subset_state = _build_repo_state_subset(repo_state, matched_paths)
    if subset_state is None:
        return None

    fine_topics = extract_fine_topics(subset_state)
    if not fine_topics:
        return f"按刚才的板块“{matched_label}”继续往下看，当前没有足够稳定的子分类可拆。"

    lines = [f"- {item['tag']}：约 {item['count']} 个文件" for item in fine_topics[:10]]
    return (
        f"按刚才的板块“{matched_label}”继续往下拆，里面主要有这些子类：\n"
        + "\n".join(lines)
        + "\n\n这是基于该板块内文件的影子标签继续归纳出来的。"
    )


def answer_repo_content_category_label_list_question(
    target_topic: str,
    repo_state,
    previous_summary: str | None,
    topic_summarizer: Callable[[str], str] | None = None,
) -> str | None:
    category_labels = _extract_summary_labels(previous_summary)
    if not category_labels:
        return None

    matched_label = _match_category_label(target_topic, category_labels)
    if not matched_label:
        matched_label = _match_category_label_with_local_llm(
            target_topic=target_topic,
            category_labels=category_labels,
            topic_summarizer=topic_summarizer,
        )
    if not matched_label:
        return None

    path_map = _assign_records_to_summary_labels_with_local_llm(
        repo_state,
        category_labels=category_labels,
        topic_summarizer=topic_summarizer,
    )
    if not path_map:
        return None

    matched_paths = [path for path, category in path_map.items() if category == matched_label]
    if not matched_paths:
        return f"按刚才的板块“{matched_label}”来看，当前没有明显归到这一类的文件。"

    lines = [f"按刚才的板块“{matched_label}”来看，相关文件大约有 {len(matched_paths)} 个，先列出这些："]
    lines.extend(f"- {path}" for path in matched_paths)
    return "\n".join(lines)
