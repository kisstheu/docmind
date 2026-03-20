from __future__ import annotations

import re


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


def extract_strong_terms_from_question(question: str) -> list[str]:
    q = (question or "").strip()
    if not q:
        return []

    candidates: list[str] = []

    candidates.extend(re.findall(r"\d{1,2}号之后", q))
    candidates.extend(re.findall(r"\d{1,2}日之后", q))
    candidates.extend(re.findall(r"\d{1,2}月\d{1,2}日", q))
    candidates.extend(re.findall(r"\d{4}年\d{1,2}月\d{1,2}日", q))
    candidates.extend(re.findall(r"[\u4e00-\u9fa5]{2,}", q))

    weak_terms = {
        "给我", "帮我", "我想", "我先", "请你",
        "分析", "整理", "梳理", "总结", "看看", "看下", "说说", "讲讲",
        "事情", "情况", "问题", "内容", "动作", "方面", "东西",
        "一下", "一下吧", "吧", "吗", "呢", "呀", "啊",
        "更详细的", "详细的", "详细点", "更详细", "详细一些",
    }

    result: list[str] = []
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        if c in weak_terms:
            continue
        if len(c) > 12:
            continue
        if c not in result:
            result.append(c)

    return result


def merge_rewritten_query_with_strong_terms(question: str, rewritten_query: str) -> str:
    rewritten_terms = [x.strip() for x in (rewritten_query or "").split() if x.strip()]
    strong_terms = extract_strong_terms_from_question(question)

    merged: list[str] = []
    for term in rewritten_terms + strong_terms:
        if term and term not in merged:
            merged.append(term)

    result = " ".join(merged).strip()
    return result or (rewritten_query or "").strip()


def is_abstract_query(question: str) -> bool:
    terms = extract_strong_terms_from_question(question)
    if not terms:
        return True

    weak_terms = {"事情", "情况", "问题", "内容", "性质"}
    return all(t in weak_terms for t in terms)


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