from __future__ import annotations

from typing import Sequence

from ai.capability_common import (
    BAD_CANDIDATE_FRAGMENTS,
    CANDIDATE_SUFFIXES,
    CATEGORY_CONFIRM_KEYWORDS,
    CATEGORY_KEYWORDS,
    CATEGORY_SUMMARY_KEYWORDS,
    CONFIRMATION_BAD_CANDIDATES,
    CONFIRMATION_STOP_PHRASES,
    LIST_FILE_KEYWORDS,
    contains_any,
    clean_text,
    dedupe_keep_order,
    extract_fine_topics,
    split_cn_text,
    summarize_topics_coarsely_with_local_llm, TOTAL_SIZE_KEYWORDS, format_bytes,
)


def is_category_summary_request(question: str) -> bool:
    return contains_any(question, CATEGORY_SUMMARY_KEYWORDS)


def is_category_confirmation_request(question: str) -> bool:
    return contains_any(question, CATEGORY_CONFIRM_KEYWORDS)


def is_followup_from_file_list(last_question: str | None, current_question: str) -> bool:
    return contains_any(last_question, LIST_FILE_KEYWORDS) and contains_any(
        current_question,
        ("方面", "分类", "类别", "哪类", "怎么分", "如何分"),
    )


def is_followup_from_category(last_question: str | None, current_question: str) -> bool:
    category_context_keywords = CATEGORY_KEYWORDS + CATEGORY_SUMMARY_KEYWORDS
    return contains_any(last_question, category_context_keywords) and (
        is_category_summary_request(current_question)
        or is_category_confirmation_request(current_question)
    )
def is_followup_to_list_files(last_topic: str | None, current_question: str) -> bool:
    return (
        last_topic in {"count", "list_files"}
        and contains_any(current_question, ("列一下", "列一下吧", "列出来", "展开一下", "展开列一下"))
    )


def calc_repo_total_bytes(repo_state) -> int:
    doc_records = getattr(repo_state, "doc_records", None) or []
    if not doc_records:
        return 0

    total_bytes = 0
    for record in doc_records:
        if not isinstance(record, dict):
            continue

        total_bytes += int(
            record.get("file_size", 0)
            or record.get("size", 0)
            or record.get("bytes", 0)
            or 0
        )

    return total_bytes

def is_followup_to_list_files(last_topic: str | None, current_question: str) -> bool:
    return (
        last_topic in {"count", "list_files"}
        and contains_any(current_question, ("列一下", "列一下吧", "列出来", "展开一下"))
    )

def classify_repo_meta_question(
    question: str,
    last_user_question: str | None = None,
    last_local_topic: str | None = None,
) -> str | None:
    q = clean_text(question)

    rules: list[tuple[str, tuple[str, ...]]] = [
        ("count", ("多少文件", "多少个文件", "文件数量", "文档数量")),
        ("total_size", TOTAL_SIZE_KEYWORDS),
        ("format", ("哪些格式", "文件格式", "文档格式", "支持格式")),
        ("time", ("最近更新", "最新文件", "最早文件")),
        ("list_files", LIST_FILE_KEYWORDS),
    ]

    for topic, keywords in rules:
        if contains_any(q, keywords):
            return topic

    if is_followup_to_list_files(last_local_topic, q):
        return "list_files"

    if is_category_summary_request(q):
        return "category_summary"

    if is_category_confirmation_request(q):
        return "category_confirm"

    if contains_any(q, CATEGORY_KEYWORDS):
        return "category"

    if is_followup_from_file_list(last_user_question, q):
        return "category"

    if is_followup_from_category(last_user_question, q):
        if is_category_summary_request(q):
            return "category_summary"
        if is_category_confirmation_request(q):
            return "category_confirm"

    return None

def answer_repo_content_category_question(repo_state) -> str:
    fine_topics = extract_fine_topics(repo_state)
    if not fine_topics:
        return "当前标签信息不足，暂时还无法按内容主题分类。"

    lines = [f"- {tag}：约 {count} 个文件" for tag, count in fine_topics[:12]]
    return (
        "按内容主题粗略来看，当前知识库主要集中在这些方面：\n"
        + "\n".join(lines)
        + "\n\n这是基于影子标签自动归纳出来的，不是按文件后缀硬分的。"
    )


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

    candidates = [
        part
        for part in parts
        if len(part) >= 2 and part not in CONFIRMATION_BAD_CANDIDATES
    ]
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
                fragment = candidate[i:i + size]
                if len(fragment) >= 2:
                    fragments.add(fragment)

    result = [
        fragment
        for fragment in fragments
        if fragment not in BAD_CANDIDATE_FRAGMENTS and len(fragment) >= 2
    ]
    result.sort(key=len, reverse=True)
    return result


def match_confirmation_candidates_to_topics(
    candidates: Sequence[str],
    fine_topics: Sequence[tuple[str, int]],
) -> list[tuple[str, str, int]]:
    matches: list[tuple[str, str, int]] = []

    for candidate in candidates:
        best_match: tuple[str, str, int] | None = None
        fragments = expand_candidate_fragments(candidate)

        for topic, count in fine_topics:
            if any(fragment in topic or topic in fragment for fragment in fragments):
                if best_match is None or count > best_match[2]:
                    best_match = (candidate, topic, count)

        if best_match:
            matches.append(best_match)

    return matches


def answer_repo_content_category_confirm_question(question: str, repo_state) -> str:
    fine_topics = extract_fine_topics(repo_state)
    if not fine_topics:
        return "当前标签信息不足，我还没法判断这个方面是不是明显更多。"

    candidates = extract_confirmation_candidates(question)
    if not candidates:
        top_preview = "、".join(name for name, _ in fine_topics[:3])
        return f"我大概明白你的意思，不过这句话里可直接比对的主题词不够明显。当前更突出的主题大致有：{top_preview}。"

    matches = match_confirmation_candidates_to_topics(candidates, fine_topics)
    if not matches:
        top_preview = "、".join(f"{name}（约{count}）" for name, count in fine_topics[:5])
        return (
            f"不一定。按当前影子标签归纳的结果，"
            f"我还看不出“{'、'.join(candidates)}”是最突出的那一类。"
            f"目前更靠前的主题有：{top_preview}。"
        )

    _best_candidate, best_topic, best_count = max(matches, key=lambda item: item[2])
    top_count = fine_topics[0][1]

    if best_count >= max(3, int(top_count * 0.7)):
        return (
            f"可以这么理解。按当前影子标签归纳结果看，"
            f"“{best_topic}”这一类确实算比较突出的主题之一，"
            f"大约对应 {best_count} 个文件。"
        )

    return (
        f"可以稍微这样理解，但没到特别压倒性的程度。"
        f"按当前结果看，和你这句话最接近的是“{best_topic}”，"
        f"大约对应 {best_count} 个文件。"
    )





def answer_repo_meta_question(
    question: str,
    repo_state,
    last_user_question: str | None = None,
    last_local_topic: str | None = None,
    topic_summarizer=None,
):
    paths = list(repo_state.paths)
    all_files = list(repo_state.all_files)
    file_times = list(repo_state.file_times)

    if not paths:
        return "当前知识库里还没有可用文档。", "empty"

    topic = classify_repo_meta_question(
        question,
        last_user_question=last_user_question,
        last_local_topic=last_local_topic,
    )
    if topic == "count":
        return f"当前知识库里共有 {len(paths)} 个可用文件。", topic
    if topic == "total_size":
        total_bytes = calc_repo_total_bytes(repo_state)
        return f"当前知识库里这些文档总共约占 {format_bytes(total_bytes)} 空间。", "total_size"
    if topic == "format":
        suffixes = sorted({file.suffix.lower() or "[无后缀]" for file in all_files})
        answer = "当前知识库中的文件格式有：\n" + "\n".join(f"- {suffix}" for suffix in suffixes)
        return answer, topic

    if topic == "time":
        latest_idx = max(range(len(file_times)), key=lambda i: file_times[i])
        earliest_idx = min(range(len(file_times)), key=lambda i: file_times[i])
        answer = (
            f"最早的文件：{paths[earliest_idx]}（{file_times[earliest_idx].strftime('%Y-%m-%d %H:%M:%S')}）\n"
            f"最新的文件：{paths[latest_idx]}（{file_times[latest_idx].strftime('%Y-%m-%d %H:%M:%S')}）"
        )
        return answer, topic

    if topic == "list_files":
        show_n = min(50, len(paths))
        preview = "\n".join(f"- {path}" for path in paths[:show_n])
        if len(paths) > show_n:
            answer = (
                f"当前知识库里共有 {len(paths)} 个文件，先列出前 {show_n} 个：\n"
                f"{preview}\n- ...(其余 {len(paths) - show_n} 个未展开)"
            )
        else:
            answer = f"当前知识库里的文件如下：\n{preview}"
        return answer, topic

    if topic == "category":
        return answer_repo_content_category_question(repo_state), topic

    if topic == "category_summary":
        return (
            answer_repo_content_category_summary_question(
                repo_state,
                topic_summarizer=topic_summarizer,
            ),
            topic,
        )

    if topic == "category_confirm":
        return answer_repo_content_category_confirm_question(question, repo_state), topic

    return None, None
