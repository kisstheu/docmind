from __future__ import annotations

import json
import re
from typing import Callable, Sequence

import numpy as np

from ai.repo_meta.category.shared import (
    _build_record_category_embedding_text,
    _build_record_category_hint,
    _extract_count_target_topic,
    _extract_summary_labels,
    _match_category_label,
    _match_category_label_with_local_llm,
    _strip_code_fence,
)
from ai.repo_meta.category.summary import answer_repo_content_category_question

_CATEGORY_ASSIGNMENT_CACHE: dict[str, dict[str, str]] = {}


def _build_category_assignment_cache_key(repo_state, category_labels: Sequence[str]) -> str:
    labels = [str(label or "").strip() for label in category_labels if str(label or "").strip()]
    records = []
    for record in list(getattr(repo_state, "doc_records", []) or []):
        path = str(record.get("path", "") or "").strip()
        hint = _build_record_category_hint(record)
        if not path or not hint:
            continue
        records.append({"path": path, "hint": hint})
    payload = {"labels": labels, "records": records}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _get_cached_category_assignment_map(
    repo_state,
    category_labels: Sequence[str],
) -> dict[str, str] | None:
    cache_key = _build_category_assignment_cache_key(repo_state, category_labels)
    cached = _CATEGORY_ASSIGNMENT_CACHE.get(cache_key)
    return dict(cached) if cached else None


def _normalize_embedding_matrix(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def _assign_records_to_summary_labels_with_embeddings(
    repo_state,
    category_labels: Sequence[str],
    model_emb,
) -> dict[str, str] | None:
    if model_emb is None:
        return None

    labels = [str(label or "").strip() for label in category_labels if str(label or "").strip()]
    if not labels:
        return None

    record_by_path: dict[str, dict] = {}
    for record in list(getattr(repo_state, "doc_records", []) or []):
        path = str(record.get("path", "") or "").strip()
        if path:
            record_by_path[path] = dict(record)

    paths: list[str] = []
    texts: list[str] = []
    for path in list(getattr(repo_state, "paths", []) or []):
        clean_path = str(path or "").strip()
        if not clean_path:
            continue
        record = dict(record_by_path.get(clean_path) or {"path": clean_path})
        text = _build_record_category_embedding_text(record)
        if not text:
            continue
        paths.append(clean_path)
        texts.append(text)

    if not texts:
        return None

    try:
        label_vecs = _normalize_embedding_matrix(model_emb.encode(labels))
        record_vecs = _normalize_embedding_matrix(model_emb.encode(texts))
    except Exception:
        return None

    if (
        label_vecs.ndim != 2
        or record_vecs.ndim != 2
        or label_vecs.shape[0] != len(labels)
        or record_vecs.shape[0] != len(texts)
    ):
        return None

    label_norms = np.linalg.norm(label_vecs, axis=1, keepdims=True)
    record_norms = np.linalg.norm(record_vecs, axis=1, keepdims=True)
    label_norms[label_norms == 0] = 1.0
    record_norms[record_norms == 0] = 1.0

    scores = (record_vecs / record_norms) @ (label_vecs / label_norms).T
    assignments: dict[str, str] = {}
    for idx, path in enumerate(paths):
        label_index = int(np.argmax(scores[idx]))
        assignments[path] = labels[label_index]
    return assignments or None


def _parse_category_assignment_output(raw_text: str, category_labels: Sequence[str]) -> dict[str, str]:
    text = _strip_code_fence(raw_text)
    allowed = {label.strip() for label in category_labels if label.strip()}
    if not text or not allowed:
        return {}

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None

    result: dict[str, str] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "") or "").strip()
            category = str(item.get("category", "") or "").strip()
            if item_id and category in allowed:
                result[item_id] = category
        if result:
            return result

    for line in text.splitlines():
        stripped = line.strip().lstrip("-*")
        match = re.match(r"(F\d+)\s*[:=：]\s*(.+)", stripped)
        if not match:
            continue
        item_id = match.group(1).strip()
        category = match.group(2).strip()
        if category in allowed:
            result[item_id] = category
    return result


def _assign_records_to_summary_labels_with_local_llm(
    repo_state,
    category_labels: Sequence[str],
    topic_summarizer: Callable[[str], str] | None,
) -> dict[str, str] | None:
    if not topic_summarizer:
        return None

    records = list(getattr(repo_state, "doc_records", []) or [])
    if not records or len(records) > 80:
        return None

    cache_key = _build_category_assignment_cache_key(repo_state, category_labels)
    cached = _CATEGORY_ASSIGNMENT_CACHE.get(cache_key)
    if cached:
        return dict(cached)

    lines: list[str] = []
    record_ids: list[str] = []
    for idx, record in enumerate(records, start=1):
        hint = _build_record_category_hint(record)
        if not hint:
            continue
        item_id = f"F{idx}"
        record_ids.append(item_id)
        lines.append(f"{item_id}: {hint}")

    if len(lines) < 2:
        return None

    prompt = (
        "下面是已经确定好的知识库粗分类板块，请不要改名：\n"
        + "\n".join(f"- {label}" for label in category_labels)
        + "\n\n下面是每个文件的标签摘要。请把每个文件归到最贴近的一个板块。\n"
        "要求：\n"
        "1. 每个文件只能归入一个板块\n"
        "2. 只能使用上面给定的板块名\n"
        "3. 如果多个板块都沾边，选最主要的那个\n"
        "4. 只输出 JSON 数组，不要解释\n"
        "5. JSON 元素格式必须是 {\"id\":\"F1\",\"category\":\"板块名\"}\n\n"
        "文件：\n"
        + "\n".join(lines)
    )

    try:
        assignments = _parse_category_assignment_output(topic_summarizer(prompt), category_labels)
    except Exception:
        return None

    if len(assignments) < max(2, int(len(record_ids) * 0.6)):
        return None

    path_map: dict[str, str] = {}
    for idx, record in enumerate(records, start=1):
        item_id = f"F{idx}"
        category = assignments.get(item_id)
        path = str(record.get("path", "") or "").strip()
        if path and category in category_labels:
            path_map[path] = category
    if len(path_map) < max(2, int(len(record_ids) * 0.6)):
        return None
    _CATEGORY_ASSIGNMENT_CACHE[cache_key] = dict(path_map)
    return path_map


def build_local_category_assignment_map(
    repo_state,
    previous_summary: str | None,
    model_emb=None,
) -> tuple[list[str], dict[str, str] | None]:
    category_labels = _extract_summary_labels(previous_summary)
    if not category_labels:
        return [], None

    cached = _get_cached_category_assignment_map(repo_state, category_labels)
    if cached:
        return category_labels, cached

    embedded = _assign_records_to_summary_labels_with_embeddings(
        repo_state,
        category_labels=category_labels,
        model_emb=model_emb,
    )
    return category_labels, embedded


def answer_repo_content_category_count_breakdown_question(
    repo_state,
    previous_summary: str | None,
    topic_summarizer: Callable[[str], str] | None = None,
) -> str:
    category_labels = _extract_summary_labels(previous_summary)
    if category_labels:
        path_map = _assign_records_to_summary_labels_with_local_llm(
            repo_state,
            category_labels=category_labels,
            topic_summarizer=topic_summarizer,
        )
        if path_map:
            counts = {label: 0 for label in category_labels}
            for category in path_map.values():
                if category in counts:
                    counts[category] += 1
            lines = [f"- {label}：约 {counts.get(label, 0)} 个文件" for label in category_labels]
            return "按刚才这些板块粗略归并后，文件数量大致是：\n" + "\n".join(lines)

    return (
        "如果你是想看更稳、可直接统计的数量，先给你细一层的分类计数：\n"
        + answer_repo_content_category_question(repo_state)
    )


def answer_repo_content_category_label_count_question(
    question: str,
    repo_state,
    previous_summary: str | None,
    model_emb=None,
    topic_summarizer: Callable[[str], str] | None = None,
) -> str | None:
    category_labels = _extract_summary_labels(previous_summary)
    if not category_labels:
        return None

    target_topic = question
    matched_label = _match_category_label(target_topic, category_labels)
    if not matched_label:
        target_topic = _extract_count_target_topic(question)
        if not target_topic:
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

    matched_count = sum(1 for category in path_map.values() if category == matched_label)
    return f"按刚才的板块“{matched_label}”来看，相关文件约有 {matched_count} 个。"
