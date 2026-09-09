from __future__ import annotations

import re


_COLLOQUIAL_REPLACEMENTS = (
    (re.compile(r"找个?仁儿"), "找人"),
    (re.compile(r"找个?仁"), "找人"),
    (re.compile(r"找个?银"), "找人"),
    (re.compile(r"找个?人儿"), "找人"),
    (re.compile(r"仁儿"), "人"),
    (re.compile(r"\b仁\b"), "人"),
    (re.compile(r"\b银\b"), "人"),
)
_CONTENT_COLLOQUIAL_PATTERN = re.compile(
    r"((?:\u8bb2|\u5199|\u8bf4|\u8bb0\u5f55|\u4ecb\u7ecd)(?:\u4e86|\u7684)?"
    r"|\u5185\u5bb9(?:\u662f|\u6709)?|\u5173\u4e8e)(?:\u5565)"
    r"(?=(?:\u7684)?[\uff1f?\u3002\uff01!\s]*$)"
)
_SENSITIVE_REDACTIONS = (
    (re.compile(r"\b\d{17}[\dXx]\b"), "[身份证号已脱敏]"),
    (re.compile(r"\b1[3-9]\d{9}\b"), "[手机号已脱敏]"),
    (
        re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b"),
        "[邮箱已脱敏]",
    ),
    (re.compile(r"\b\d{16,19}\b"), "[长数字已脱敏]"),
)
_STRUCTURED_REQUEST_PREFIX_PATTERN = re.compile(
    r"^(给我|帮我|请你|麻烦你|我想|我先|先)\s*"
)
_STRUCTURED_REQUEST_SUFFIX_PATTERN = re.compile(r"(吧|吗|呢|呀|啊)$")
_STRUCTURED_REQUEST_WORDS_PATTERN = re.compile(
    r"(时间线|时间顺序|整理一下|梳理一下|分析一下|总结一下)"
)
_WHITESPACE_PATTERN = re.compile(r"\s+")


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
_SORTED_QUERY_FILLERS = tuple(sorted(QUERY_FILLERS, key=len, reverse=True))
_STRONG_DATE_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\d{4}年\d{1,2}月\d{1,2}[日号]?(?:后|之前|之后|以后)?",
        r"\d{1,2}月\d{1,2}[日号]?(?:后|之前|之后|以后)?",
        r"\d{1,2}[日号](?:后|之前|之后|以后)?",
        r"\d{1,2}:\d{2}",
    )
)
_STRUCTURAL_QUERY_TERMS = (
    "时间线",
    "经过",
    "过程",
    "之后",
    "后来",
    "后续",
)
_TIMELINE_DATE_PATTERN = re.compile(
    r"(?:\d{4}年\d{1,2}月\d{1,2}日|\d{1,2}月\d{1,2}日|\d{1,2}日|\d{1,2}:\d{2})"
)
_TIMELINE_EVIDENCE_KEYWORDS = ("时间线", "经过", "过程", "梳理", "更详细", "详细点")
_RESULT_EXPANSION_MARKERS = {
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
_ANSWER_DEPTH_FOLLOWUP_PATTERNS = (
    re.compile(
        r"^(?:\u53ef\u4ee5|\u80fd\u5426|\u80fd\u4e0d\u80fd|\u8bf7|\u9ebb\u70e6)?"
        r"(?:\u518d|\u66f4)?"
        r"(?:(?:\u5177\u4f53|\u8be6\u7ec6|\u6df1\u5165)"
        r"(?:(?:\u8bf4|\u8bb2)(?:\u8bf4|\u8bb2)?|\u8bf4\u660e|\u5206\u6790|\u89e3\u91ca)?"
        r"|\u5c55\u5f00(?:(?:\u8bf4|\u8bb2)(?:\u8bf4|\u8bb2)?)?)"
        r"(?:\u4e00\u4e0b|\u4e0b|\u4e9b|\u4e00\u70b9|\u4e00\u4e9b|\u70b9|\u4e00\u70b9\u70b9)?"
        r"(?:\u5417|\u5462|\u5427)?$"
    ),
    re.compile(
        r"^(?:\u53ef\u4ee5|\u80fd\u5426|\u80fd\u4e0d\u80fd)?(?:\u518d)?"
        r"(?:\u8bf4|\u8bb2|\u56de\u7b54)(?:\u5f97)?(?:\u518d|\u66f4)?"
        r"(?:\u5177\u4f53|\u8be6\u7ec6|\u6df1\u5165)"
        r"(?:\u4e00\u4e0b|\u4e0b|\u4e9b|\u4e00\u70b9|\u4e00\u4e9b|\u70b9|\u4e00\u70b9\u70b9)?"
        r"(?:\u5417|\u5462|\u5427)?$"
    ),
)
_RELATED_MARKERS = ("有关", "相关")
_RECORD_SCOPE_MARKERS = ("记录", "文档", "文件")
_LISTING_MARKERS = ("哪些", "哪几", "有哪", "最近")


def _normalize_spaces(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def normalize_colloquial_question(question: str) -> str:
    q = question.strip()

    for pattern, repl in _COLLOQUIAL_REPLACEMENTS:
        q = pattern.sub(repl, q)

    q = _CONTENT_COLLOQUIAL_PATTERN.sub(
        lambda match: f"{match.group(1)}什么",
        q,
    )

    return q


def redact_sensitive_text(text: str) -> str:
    t = text or ""
    for pattern, replacement in _SENSITIVE_REDACTIONS:
        t = pattern.sub(replacement, t)
    return t


def strip_structured_request_words(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    t = _STRUCTURED_REQUEST_PREFIX_PATTERN.sub("", t)
    t = _STRUCTURED_REQUEST_SUFFIX_PATTERN.sub("", t)
    t = _STRUCTURED_REQUEST_WORDS_PATTERN.sub(" ", t)

    if t in {"更详细的", "详细的", "详细点", "更详细", "详细一些"}:
        return ""

    return _normalize_spaces(t)


def build_clean_merged_query(event_merged_query: str, current_question: str) -> str:
    parent = strip_structured_request_words(event_merged_query)
    current = strip_structured_request_words(current_question)

    if not parent and not current:
        return (current_question or "").strip()
    if not parent:
        return current or (current_question or "").strip()
    if not current:
        return parent

    return _normalize_spaces(f"{parent} {current}")


def normalize_question_for_retrieval(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""

    q = q.replace("？", "").replace("?", "").replace("。", "").strip()

    for filler in _SORTED_QUERY_FILLERS:
        q = q.replace(filler, " ")

    return _normalize_spaces(q)


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


def _extract_strong_terms_from_normalized_question(q: str) -> list[str]:
    result: list[str] = []

    def add(term: str):
        t = (term or "").strip()
        if t and t not in result:
            result.append(t)

    # 只提取问题里明确写出来的时间短语
    for pattern in _STRONG_DATE_PATTERNS:
        for match in pattern.findall(q):
            add(match)

    # Core 只提升通用结构语义，不维护业务或普通内容词表。
    for term in _STRUCTURAL_QUERY_TERMS:
        if term in q:
            add(term)

    return result


def extract_strong_terms_from_question(question: str) -> list[str]:
    q = normalize_question_for_retrieval(question)
    if not q:
        return []
    return _extract_strong_terms_from_normalized_question(q)


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

    terms = _extract_strong_terms_from_normalized_question(q)
    return not terms


def extract_timeline_evidence_from_chunks(relevant_indices, repo_state):
    results = []
    seen = set()
    for idx in relevant_indices:
        text = repo_state.chunk_texts[idx]
        path = repo_state.chunk_paths[idx]

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            key = (path, line)
            if _TIMELINE_DATE_PATTERN.search(line) and key not in seen:
                seen.add(key)
                results.append(key)

    return results


def build_timeline_evidence_text(timeline_items):
    if not timeline_items:
        return ""

    lines = [f"{path} | {line}" for path, line in timeline_items[:80]]
    return "\n".join(lines) + "\n\n"


def needs_timeline_evidence(question: str) -> bool:
    return any(x in question for x in _TIMELINE_EVIDENCE_KEYWORDS)


def is_result_expansion_followup(question: str) -> bool:
    if is_answer_depth_followup(question):
        return True

    raw_q = (question or "").strip()
    if "详细点" in raw_q:
        return True

    q = normalize_question_for_retrieval(question)
    if not q:
        return False

    return any(x in q for x in _RESULT_EXPANSION_MARKERS)


def is_answer_depth_followup(question: str) -> bool:
    q = re.sub(r"[，。！？、,.!?；;：:\s]+", "", (question or "").strip().lower())
    if not q or len(q) > 24:
        return False
    return any(pattern.fullmatch(q) for pattern in _ANSWER_DEPTH_FOLLOWUP_PATTERNS)


def is_related_record_listing_request(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    has_related = any(x in q for x in _RELATED_MARKERS)
    has_record_scope = any(x in q for x in _RECORD_SCOPE_MARKERS)
    has_listing = any(x in q for x in _LISTING_MARKERS)
    return has_related and has_record_scope and has_listing
