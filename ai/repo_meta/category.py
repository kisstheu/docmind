from __future__ import annotations

from typing import Sequence

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
