from __future__ import annotations

import re
from pathlib import Path


def normalize_colloquial_question(question: str) -> str:
    q = question.strip()

    replacements = [
        (r"找个?仁儿", "找人"),
        (r"找个?仁", "找人"),
        (r"找个?银", "找人"),
        (r"找个?人儿", "找人"),
        (r"仁儿", "人"),
        (r"\b仁\b", "人"),
        (r"\b银\b", "人"),
    ]

    for pattern, repl in replacements:
        q = re.sub(pattern, repl, q)

    return q


def redact_sensitive_text(text: str) -> str:
    t = text or ""
    t = re.sub(r"\b\d{17}[\dXx]\b", "[身份证号已脱敏]", t)
    t = re.sub(r"\b1[3-9]\d{9}\b", "[手机号已脱敏]", t)
    t = re.sub(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "[邮箱已脱敏]", t)
    t = re.sub(r"\b\d{16,19}\b", "[长数字已脱敏]", t)
    return t


def strip_structured_request_words(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    t = re.sub(r"^(给我|帮我|请你|麻烦你|我想|我先|先)\s*", "", t)
    t = re.sub(r"(吧|吗|呢|呀|啊)$", "", t)
    t = re.sub(r"(时间线|时间顺序|整理一下|梳理一下|分析一下|总结一下)", " ", t)

    if t in {"更详细的", "详细的", "详细点", "更详细", "详细一些"}:
        return ""

    return re.sub(r"\s+", " ", t).strip()


def build_clean_merged_query(event_merged_query: str, current_question: str) -> str:
    parent = strip_structured_request_words(event_merged_query)
    current = strip_structured_request_words(current_question)

    if not parent and not current:
        return (current_question or "").strip()
    if not parent:
        return current or (current_question or "").strip()
    if not current:
        return parent

    return re.sub(r"\s+", " ", f"{parent} {current}".strip())


QUERY_FILLERS = {
    "帮我",
    "帮忙",
    "请",
    "麻烦",
    "看下",
    "看看",
    "分析下",
    "分析一下",
    "整理下",
    "整理一下",
    "说下",
    "说一下",
    "详细点",
    "具体点",
    "展开点",
    "展开说说",
}


def normalize_question_for_retrieval(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""

    q = q.replace("？", "").replace("?", "").replace("。", "").strip()

    for filler in sorted(QUERY_FILLERS, key=len, reverse=True):
        q = q.replace(filler, " ")

    q = re.sub(r"\s+", " ", q).strip()
    return q


def keep_only_allowed_terms(query: str, question: str, logger=None) -> str:
    """
    只保留“当前问题里本来就出现过”的词。
    rewrite 可以重排，但不允许新增词。
    """
    source_text = normalize_question_for_retrieval(question) or (question or "").strip()

    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()

    for term in (query or "").split():
        t = term.strip()
        if not t or t in seen:
            continue

        if t in source_text:
            kept.append(t)
            seen.add(t)
        else:
            dropped.append(t)

    if dropped and logger:
        logger.info(f"🚫 [过滤新增词] {dropped}")

    return " ".join(kept)


def extract_strong_terms_from_question(question: str) -> list[str]:
    q = normalize_question_for_retrieval(question)
    if not q:
        return []

    result: list[str] = []

    def add(term: str):
        t = (term or "").strip()
        if t and t not in result:
            result.append(t)

    # 只提取问题里明确写出来的时间短语
    for pattern in [
        r"\d{4}年\d{1,2}月\d{1,2}[日号]?(?:后|之前|之后|以后)?",
        r"\d{1,2}月\d{1,2}[日号]?(?:后|之前|之后|以后)?",
        r"\d{1,2}[日号](?:后|之前|之后|以后)?",
        r"\d{1,2}:\d{2}",
    ]:
        for match in re.findall(pattern, q):
            add(match)

    # 只保留“当前问题中原样出现”的少量焦点词
    exact_terms = [
        "时间线",
        "经过",
        "过程",
        "详细点",
        "更详细",
        "法律性质",
        "性质",
        "合法吗",
        "是否合法",
        "合法",
        "合规吗",
        "合规",
        "动作",
        "做法",
        "行为",
        "处理",
        "公司",
        "对方",
        "之后",
        "后来",
        "后续",
    ]

    for term in exact_terms:
        if term in q:
            add(term)

    return result


def merge_rewritten_query_with_strong_terms(question: str, rewritten_query: str, logger=None) -> str:
    # rewrite 只能重排当前问题已有的词，不允许新增
    safe_rewritten = keep_only_allowed_terms(
        rewritten_query,
        question,
        logger=logger,
    )

    rewritten_terms = [x.strip() for x in safe_rewritten.split() if x.strip()]
    strong_terms = extract_strong_terms_from_question(question)

    merged: list[str] = []
    for term in rewritten_terms + strong_terms:
        if term and term not in merged:
            merged.append(term)

    result = " ".join(merged).strip()
    return result or (normalize_question_for_retrieval(question) or (question or "").strip())


def is_abstract_query(question: str) -> bool:
    q = normalize_question_for_retrieval(question)
    if not q:
        return True

    terms = extract_strong_terms_from_question(q)
    return not terms


def extract_timeline_evidence_from_chunks(relevant_indices, repo_state):
    date_patterns = [
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d{1,2}月\d{1,2}日",
        r"\d{1,2}日",
        r"\d{1,2}:\d{2}",
    ]

    results = []
    for idx in relevant_indices:
        text = repo_state.chunk_texts[idx]
        path = repo_state.chunk_paths[idx]

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            if any(re.search(p, line) for p in date_patterns):
                results.append((path, line))

    seen = set()
    deduped = []
    for path, line in results:
        key = (path, line)
        if key not in seen:
            seen.add(key)
            deduped.append((path, line))

    return deduped


def build_timeline_evidence_text(timeline_items):
    if not timeline_items:
        return ""

    lines = [f"{path} | {line}" for path, line in timeline_items[:80]]
    return "\n".join(lines) + "\n\n"


def needs_timeline_evidence(question: str) -> bool:
    keywords = ["时间线", "经过", "过程", "梳理", "更详细", "详细点"]
    return any(x in question for x in keywords)


def is_result_expansion_followup(question: str) -> bool:
    q = normalize_question_for_retrieval(question)
    if not q:
        return False

    markers = {
        "更详细",
        "更详细的",
        "详细点",
        "具体点",
        "展开点",
        "展开说说",
        "继续",
        "然后呢",
        "后来呢",
        "扩大范围",
        "范围大点",
        "范围放宽",
        "放宽范围",
        "放宽一点",
        "扩大检索",
        "分析下",
        "分析一下",
        "法律性质",
        "性质",
        "合法吗",
        "是否合法",
    }

    return any(x in q for x in markers)


def is_related_record_listing_request(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    has_related = any(x in q for x in ["有关", "相关"])
    has_record_scope = any(x in q for x in ["记录", "文档", "文件"])
    has_listing = any(x in q for x in ["哪些", "哪几", "有哪", "最近"])
    return has_related and has_record_scope and has_listing
