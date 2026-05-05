from __future__ import annotations

import re
from typing import Callable

from ai.capability_common import extract_fine_topics


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
