from __future__ import annotations

import re
from typing import Callable, Sequence

from ai.capability_common import (
    BAD_CANDIDATE_FRAGMENTS,
    CANDIDATE_SUFFIXES,
    CONFIRMATION_BAD_CANDIDATES,
    CONFIRMATION_STOP_PHRASES,
    clean_text,
    dedupe_keep_order,
    extract_fine_topics,
    split_cn_text,
    summarize_topics_coarsely_with_local_llm,
)



def answer_repo_content_category_question(repo_state) -> str:
    fine_topics = extract_fine_topics(repo_state)
    lines = [f"- {item['tag']}：约 {item['count']} 个文件" for item in fine_topics[:12]]
    return "按内容标签粗略来看，当前知识库主要集中在这些方面：\n" + "\n".join(lines) + "\n\n这是基于影子标签自动归纳出来的。"



def answer_repo_content_category_summary_question(repo_state, topic_summarizer) -> str:
    fine_topics = extract_fine_topics(repo_state)
    if not fine_topics:
        return "当前标签信息不足，暂时还无法按更大的方面概括。"

    return summarize_topics_coarsely_with_local_llm(
        fine_topics=fine_topics,
        topic_summarizer=topic_summarizer,
    )


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


def _pick_overview_topics(repo_state) -> tuple[list[dict], str]:
    scene_topics = _extract_scene_topics(repo_state)
    if scene_topics:
        return scene_topics, "场景标签"
    return extract_fine_topics(repo_state), "细标签"


def _build_dominant_topic_overview(repo_state, topics: list[dict], topic_source: str) -> str | None:
    if topic_source != "场景标签" or not topics:
        return None

    total_docs = len(getattr(repo_state, "paths", []) or [])
    if total_docs <= 0:
        total_docs = len(getattr(repo_state, "doc_records", []) or [])
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
