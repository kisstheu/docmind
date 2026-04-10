from __future__ import annotations

import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Callable, Sequence

import numpy as np

from ai.capability_common import (
    BAD_CANDIDATE_FRAGMENTS,
    CANDIDATE_SUFFIXES,
    CONFIRMATION_BAD_CANDIDATES,
    CONFIRMATION_STOP_PHRASES,
    clean_text,
    dedupe_keep_order,
    extract_fine_topics,
    normalize_meta_question,
    split_cn_text,
    summarize_topics_coarsely_with_local_llm,
)

_CATEGORY_ASSIGNMENT_CACHE: dict[str, dict[str, str]] = {}
COUNT_TARGET_NOISE_PHRASES = (
    "有多少个文件",
    "有多少个文档",
    "有多少文件",
    "有多少文档",
    "多少个文件",
    "多少个文档",
    "多少文件",
    "多少文档",
    "文件数量",
    "文档数量",
    "文件数",
    "文档数",
    "当前",
    "目前",
    "现在",
    "总共",
    "一共",
    "总计",
    "大概",
    "大约",
    "差不多",
)
GENERIC_COUNT_TARGETS = {
    "这块",
    "这一块",
    "这类",
    "这一类",
    "这些",
    "那些",
    "这个",
    "那个",
    "这里",
    "这边",
    "那边",
    "本地",
    "仓库",
    "知识库",
}



def answer_repo_content_category_question(repo_state) -> str:
    fine_topics = extract_fine_topics(repo_state)
    lines = [f"- {item['tag']}：约 {item['count']} 个文件" for item in fine_topics[:12]]
    return "按内容标签粗略来看，当前知识库主要集中在这些方面：\n" + "\n".join(lines) + "\n\n这是基于影子标签自动归纳出来的。"



def answer_repo_content_category_summary_question(repo_state, topic_summarizer) -> str:
    scene_summary = _build_scene_category_summary(repo_state)
    if scene_summary:
        return scene_summary

    scene_topics = _extract_scene_topics(repo_state)
    if scene_topics:
        try:
            return summarize_topics_coarsely_with_local_llm(
                fine_topics=scene_topics,
                topic_summarizer=topic_summarizer,
                topic_source="场景标签",
                prefer_scene=True,
            )
        except Exception:
            pass

    fine_topics = extract_fine_topics(repo_state)
    if not fine_topics:
        return "当前标签信息不足，暂时还无法按更大的方面概括。"

    return summarize_topics_coarsely_with_local_llm(
        fine_topics=fine_topics,
        topic_summarizer=topic_summarizer,
        topic_source="细标签",
    )


def _extract_summary_labels(summary_text: str | None) -> list[str]:
    labels: list[str] = []
    for line in (summary_text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        label = stripped[2:].strip()
        label = re.sub(r"[：:]\s*约\s*\d+\s*个文件$", "", label).strip()
        if label:
            labels.append(label)
    return labels


def _extract_count_target_topic(question: str) -> str:
    target = normalize_meta_question(clean_text(question))
    if not target:
        return ""

    for phrase in sorted(COUNT_TARGET_NOISE_PHRASES, key=len, reverse=True):
        target = target.replace(phrase, " ")

    target = re.sub(r"[？?，,。.!！、；;：:\s]+", " ", target)
    target = target.strip(" 的")
    target = re.sub(r"(呢|啊|呀|吗|么|吧)$", "", target).strip()
    if not target or target in GENERIC_COUNT_TARGETS:
        return ""
    return target


def _match_category_label(target_topic: str, category_labels: Sequence[str]) -> str | None:
    target = str(target_topic or "").strip()
    if not target:
        return None

    simplified_target = target.replace("相关", "").replace("类别", "").replace("板块", "").strip()
    for label in sorted(category_labels, key=len, reverse=True):
        normalized_label = str(label or "").strip()
        if not normalized_label:
            continue
        simplified_label = normalized_label.replace("相关", "").replace("类别", "").replace("板块", "").strip()
        if target == normalized_label or target in normalized_label or normalized_label in target:
            return normalized_label
        if simplified_target and simplified_label and (
            simplified_target == simplified_label
            or simplified_target in simplified_label
            or simplified_label in simplified_target
        ):
            return normalized_label
    return None


def _match_category_label_with_local_llm(
    target_topic: str,
    category_labels: Sequence[str],
    topic_summarizer: Callable[[str], str] | None,
) -> str | None:
    if not topic_summarizer:
        return None

    labels = [str(label or "").strip() for label in category_labels if str(label or "").strip()]
    if not labels:
        return None

    prompt = (
        "下面是知识库里已经确定好的板块名，请从中选出和用户说法最接近的一个。\n"
        "要求：\n"
        "1. 只能输出候选里的原词\n"
        "2. 如果没有明显对应，也只输出最接近的一个候选\n"
        "3. 不要解释，不要输出其他内容\n\n"
        "候选板块：\n"
        + "\n".join(f"- {label}" for label in labels)
        + f"\n\n用户说法：{target_topic}"
    )

    try:
        raw = topic_summarizer(prompt)
    except Exception:
        return None

    text = _strip_code_fence(raw).strip()
    text = text.splitlines()[0].strip().lstrip("-*0123456789.、 ").strip()
    text = text.strip("[](){}<>【】「」『』“”\"'` ")
    if text in labels:
        return text
    return _match_category_label(text, labels)


def _trim_topic_text(text: str, max_items: int = 5) -> str:
    parts = [part.strip() for part in str(text or "").split() if part.strip()]
    return " ".join(parts[:max_items])


def _build_record_category_hint(record: dict) -> str:
    scene = _trim_topic_text(record.get("scene_tags", ""), max_items=4)
    shadow = _trim_topic_text(record.get("shadow_tags", ""), max_items=5)
    chunks: list[str] = []
    if scene:
        chunks.append(f"场景={scene}")
    if shadow:
        chunks.append(f"特征={shadow}")
    return "；".join(chunks)


def _build_record_category_embedding_text(record: dict) -> str:
    path = str(record.get("path", "") or "").strip()
    stem = Path(path).stem.replace("_", " ").replace("-", " ").strip() if path else ""
    hint = _build_record_category_hint(record)
    parts = [part for part in (stem, hint) if part]
    return " ".join(parts).strip()


def _strip_code_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


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


def _normalize_overview_sentence(raw_text: str) -> str:
    lines = [line.strip() for line in (raw_text or "").splitlines() if line.strip()]
    if not lines:
        return ""

    first_line = lines[0].lstrip("-*0123456789.、 ").strip()
    if not first_line:
        return ""

    normalized_line = first_line.replace("这个知识库", "这些文档").replace("知识库", "文档集合")
    if any(x in normalized_line for x in ("这些文档", "文档集合", "主要是")):
        return normalized_line if normalized_line.endswith("。") else f"{normalized_line}。"

    normalized = normalized_line.rstrip("。")
    if normalized.startswith("关于"):
        normalized = normalized[2:]
    return f"整体看，这些文档主要围绕{normalized}。"


def _format_weighted_topics(fine_topics, limit: int = 12) -> str:
    lines = []
    for item in fine_topics[:limit]:
        tag = (item.get("tag") or "").strip()
        count = int(item.get("count", 0) or 0)
        if not tag:
            continue
        lines.append(f"- {tag}（约 {count}）")
    return "\n".join(lines)


def _extract_scene_topics(repo_state) -> list[dict]:
    path_map: dict[str, set[str]] = {}
    for record in getattr(repo_state, "doc_records", []):
        path = str(record.get("path", "") or "")
        raw_scene_tags = str(record.get("scene_tags", "") or "").strip()
        if not path or not raw_scene_tags:
            continue

        for token in re.split(r"[\s,，、;；|]+", raw_scene_tags):
            tag = token.strip()
            if not tag:
                continue
            if len(tag) < 2 or len(tag) > 20:
                continue
            if tag not in path_map:
                path_map[tag] = set()
            path_map[tag].add(path)

    results = [
        {"tag": tag, "count": len(paths), "paths": sorted(paths)}
        for tag, paths in path_map.items()
    ]
    results.sort(key=lambda item: item["count"], reverse=True)
    return results


def _count_total_docs(repo_state) -> int:
    total_docs = len(getattr(repo_state, "paths", []) or [])
    if total_docs <= 0:
        total_docs = len(getattr(repo_state, "doc_records", []) or [])
    return total_docs


def _should_prefer_scene_topics_for_summary(repo_state, scene_topics: list[dict]) -> bool:
    total_docs = _count_total_docs(repo_state)
    if total_docs <= 0 or not scene_topics:
        return False

    covered_paths: set[str] = set()
    for item in scene_topics:
        covered_paths.update(str(path) for path in item.get("paths", []) if path)

    if covered_paths and (len(covered_paths) / float(total_docs)) < 0.55:
        return False

    top_count = int(scene_topics[0].get("count", 0) or 0)
    if (top_count / float(total_docs)) >= 0.45:
        return True

    strong_threshold = max(2, int(total_docs * 0.2))
    strong_topics = sum(
        1
        for item in scene_topics[:5]
        if int(item.get("count", 0) or 0) >= strong_threshold
    )
    return strong_topics >= 3


def _build_scene_category_summary(repo_state) -> str | None:
    scene_topics = _extract_scene_topics(repo_state)
    if not _should_prefer_scene_topics_for_summary(repo_state, scene_topics):
        return None

    total_docs = _count_total_docs(repo_state)
    min_count = 1 if total_docs <= 3 else 2
    selected_tags: list[str] = []
    for item in scene_topics:
        tag = str(item.get("tag", "") or "").strip()
        count = int(item.get("count", 0) or 0)
        if not tag or count < min_count:
            continue
        selected_tags.append(tag)
        if len(selected_tags) >= 5:
            break

    if not selected_tags:
        top_tag = str(scene_topics[0].get("tag", "") or "").strip()
        if not top_tag:
            return None
        selected_tags = [top_tag]

    lines = "\n".join(f"- {tag}" for tag in selected_tags)
    return "按更大的方面看，当前知识库主要集中在这些板块：\n" + lines


def _pick_overview_topics(repo_state) -> tuple[list[dict], str]:
    scene_topics = _extract_scene_topics(repo_state)
    if scene_topics:
        return scene_topics, "场景标签"
    return extract_fine_topics(repo_state), "细标签"


def _build_dominant_topic_overview(repo_state, topics: list[dict], topic_source: str) -> str | None:
    if topic_source != "场景标签" or not topics:
        return None

    total_docs = _count_total_docs(repo_state)
    if total_docs <= 0:
        return None

    top = topics[0]
    top_tag = str(top.get("tag", "") or "").strip()
    top_count = int(top.get("count", 0) or 0)
    if not top_tag or top_count <= 0:
        return None

    coverage = top_count / float(total_docs)
    if coverage < 0.55:
        return None

    secondary_tags: list[str] = []
    threshold = max(2, int(total_docs * 0.15))
    for item in topics[1:5]:
        tag = str(item.get("tag", "") or "").strip()
        count = int(item.get("count", 0) or 0)
        if not tag or count < threshold:
            continue
        secondary_tags.append(tag)

    if secondary_tags:
        if len(secondary_tags) >= 2:
            return f"整体看，这些文档主要是{top_tag}，并围绕{secondary_tags[0]}、{secondary_tags[1]}等方向展开。"
        return f"整体看，这些文档主要是{top_tag}，并围绕{secondary_tags[0]}等方向展开。"
    return f"整体看，这些文档主要是{top_tag}。"


def _summarize_category_overview_with_local_llm(
    fine_topics,
    topic_summarizer: Callable[[str], str] | None,
    previous_summary: str | None = None,
    topic_source: str = "标签",
) -> str | None:
    if not topic_summarizer:
        return None

    weighted_topics = _format_weighted_topics(fine_topics, limit=12)
    if not weighted_topics:
        return None

    previous_block = ""
    if previous_summary and previous_summary.strip():
        previous_block = (
            "上一轮较粗粒度摘要（可参考，不必照搬）：\n"
            f"{previous_summary.strip()}\n\n"
        )

    prompt = (
        "下面是个人知识库中出现频率较高的一批标签。\n"
        f"标签来源：{topic_source}。\n"
        "请再向上抽象一层，用一句中文概括整个知识库最核心的主线。\n"
        "优先概括“资料用途/问题场景”，不要只停留在技术名词罗列。\n"
        "如果技术词和场景词同时出现，优先保留场景主线。\n"
        "必须遵守：\n"
        "1. 只输出一句话\n"
        "2. 不要列表，不要解释，不要复述全部标签\n"
        "3. 表达层级尽量接近“这些文档整体主要是关于 xxx”\n\n"
        + previous_block
        + "标签：\n"
        + weighted_topics
    )

    try:
        raw_text = topic_summarizer(prompt)
    except Exception:
        return None

    normalized = _normalize_overview_sentence(raw_text)
    return normalized or None


def answer_repo_content_category_overview_question(
    repo_state,
    topic_summarizer=None,
    previous_summary: str | None = None,
) -> str:
    fine_topics, topic_source = _pick_overview_topics(repo_state)
    if not fine_topics:
        return "当前标签信息不足，暂时还无法继续向上概括。"

    dominant_overview = _build_dominant_topic_overview(repo_state, fine_topics, topic_source)
    if dominant_overview:
        return dominant_overview

    llm_overview = _summarize_category_overview_with_local_llm(
        fine_topics=fine_topics,
        topic_summarizer=topic_summarizer,
        previous_summary=previous_summary,
        topic_source=topic_source,
    )
    if llm_overview:
        return llm_overview

    top_tags = [item["tag"] for item in fine_topics[:2]]
    if len(top_tags) == 1:
        return f"整体看，这些文档主要围绕{top_tags[0]}。"
    return f"整体看，这些文档主要围绕{top_tags[0]}和{top_tags[1]}。"



def extract_confirmation_candidates(question: str) -> list[str]:
    q = clean_text(question)
    if not q:
        return []

    for phrase in CONFIRMATION_STOP_PHRASES:
        q = q.replace(phrase, " ")

    raw_parts = split_cn_text(q)
    parts: list[str] = []
    for part in raw_parts:
        for segment in part.replace("的", " ").replace("了", " ").split():
            segment = segment.strip()
            if segment:
                parts.append(segment)

    candidates = [part for part in parts if len(part) >= 2 and part not in CONFIRMATION_BAD_CANDIDATES]
    return dedupe_keep_order(candidates)



def expand_candidate_fragments(candidate: str) -> list[str]:
    candidate = candidate.strip()
    if not candidate:
        return []

    fragments = {candidate}

    for suffix in CANDIDATE_SUFFIXES:
        if candidate.endswith(suffix) and len(candidate) > len(suffix) + 1:
            fragments.add(candidate[:-len(suffix)])

    candidate_len = len(candidate)
    if 4 <= candidate_len <= 6:
        for size in range(2, min(4, candidate_len) + 1):
            for i in range(0, candidate_len - size + 1):
                fragment = candidate[i : i + size]
                if len(fragment) >= 2:
                    fragments.add(fragment)

    result = [fragment for fragment in fragments if fragment not in BAD_CANDIDATE_FRAGMENTS and len(fragment) >= 2]
    result.sort(key=len, reverse=True)
    return result



def match_confirmation_candidates_to_topics(candidates: Sequence[str], fine_topics) -> list[tuple[str, str, int]]:
    matches: list[tuple[str, str, int]] = []

    for candidate in candidates:
        best_match: tuple[str, str, int] | None = None
        fragments = expand_candidate_fragments(candidate)

        for item in fine_topics:
            tag = item["tag"]
            count = item["count"]
            if any(fragment in tag or tag in fragment for fragment in fragments):
                if best_match is None or count > best_match[2]:
                    best_match = (candidate, tag, count)

        if best_match:
            matches.append(best_match)

    return matches



def answer_repo_content_category_confirm_question(question: str, repo_state) -> str:
    fine_topics = extract_fine_topics(repo_state)
    if not fine_topics:
        return "当前标签信息不足，我还没法判断这个方面是不是明显更多。"

    candidates = extract_confirmation_candidates(question)
    if not candidates:
        top_preview = "、".join(item["tag"] for item in fine_topics[:3])
        return f"我大概明白你的意思，不过这句话里可直接比对的主题词不够明显。当前更突出的主题大致有：{top_preview}。"

    matches = match_confirmation_candidates_to_topics(candidates, fine_topics)
    if not matches:
        top_preview = "、".join(f"{item['tag']}（约{item['count']}）" for item in fine_topics[:5])
        return (
            f"不一定。按当前影子标签归纳的结果，我还看不出“{'、'.join(candidates)}”是最突出的那一类。"
            f"目前更靠前的主题有：{top_preview}。"
        )

    _best_candidate, best_topic, best_count = max(matches, key=lambda item: item[2])
    top_count = fine_topics[0]["count"]

    if best_count >= max(3, int(top_count * 0.7)):
        return (
            f"可以这么理解。按当前影子标签归纳结果看，“{best_topic}”这一类确实算比较突出的主题之一，"
            f"大约对应 {best_count} 个文件。"
        )

    return (
        f"可以稍微这样理解，但没到特别压倒性的程度。按当前结果看，和你这句话最接近的是“{best_topic}”，"
        f"大约对应 {best_count} 个文件。"
    )
