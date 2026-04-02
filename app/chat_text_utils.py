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


FILE_LOOKUP_POLITE_PREFIXES = (
    "帮我查下",
    "帮我查找",
    "帮我查询",
    "帮我找下",
    "帮我看看",
    "帮我看下",
    "帮我定位下",
    "帮我定位一下",
    "帮我",
    "麻烦你",
    "麻烦",
    "请你",
    "请",
    "查下",
    "查找",
    "查询",
    "看看",
    "看下",
    "定位下",
    "定位一下",
)

FILE_LOOKUP_GENERIC_TERMS = {
    "文件",
    "文档",
    "记录",
    "哪个",
    "哪些",
    "哪份",
    "哪一份",
    "在哪",
    "在哪个",
    "是在",
    "是在哪",
    "文件名",
    "文档名",
    "记录名",
    "帮我",
    "查下",
    "查找",
    "查询",
    "看看",
    "看下",
}


def _normalize_lookup_token(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "", (text or "").lower())


def _strip_file_lookup_prefix(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""

    changed = True
    while changed and q:
        changed = False
        for prefix in FILE_LOOKUP_POLITE_PREFIXES:
            if q.startswith(prefix):
                q = q[len(prefix):].strip(" ，,：:")
                changed = True
                break
    return q


def _extract_file_lookup_target(question: str) -> str:
    q = _strip_file_lookup_prefix(question)
    if not q:
        return ""

    compact = re.sub(r"\s+", "", q)
    m = re.search(
        r"(?P<target>.+?)(?:是)?(?:在)?(?:哪(?:个|些|份)?(?:文件|文档|记录)|哪个(?:文件|文档|记录))",
        compact,
    )
    if not m:
        return ""

    target = (m.group("target") or "").strip(" ，,：:。！？!?")
    if not target:
        return ""

    for prefix in FILE_LOOKUP_POLITE_PREFIXES:
        if target.startswith(prefix):
            target = target[len(prefix):].strip(" ，,：:")
    return target


def _build_file_lookup_terms(question: str, search_query: str, target: str) -> list[str]:
    # Local import to keep this module independent from retrieval bootstrap order.
    from retrieval.query_utils import extract_query_terms

    raw_terms = list(extract_query_terms(search_query or "", question or ""))
    if target:
        raw_terms.insert(0, target)
        target_norm = _normalize_lookup_token(target)
        if target_norm and target_norm != target:
            raw_terms.insert(1, target_norm)

        for token in re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", target.lower()):
            raw_terms.append(token)

    terms: list[str] = []
    seen: set[str] = set()
    for raw in raw_terms:
        t = (raw or "").strip()
        if not t:
            continue
        t_norm = _normalize_lookup_token(t)
        if not t_norm:
            continue
        if t_norm in FILE_LOOKUP_GENERIC_TERMS:
            continue
        if t_norm.startswith(("帮我", "请", "麻烦")):
            continue
        if ("文件" in t_norm or "文档" in t_norm or "记录" in t_norm) and ("哪" in t_norm or "在" in t_norm):
            continue
        if len(t_norm) <= 1:
            continue
        if t_norm in seen:
            continue
        seen.add(t_norm)
        terms.append(t)
    return terms


def maybe_build_file_location_answer(
    *,
    question: str,
    search_query: str,
    relevant_indices,
    repo_state,
    max_items: int = 5,
    logger=None,
) -> str | None:
    # Local import to avoid any potential cross-module initialization coupling.
    from retrieval.search_intent import is_file_location_lookup_query

    if not relevant_indices:
        return None

    if not is_file_location_lookup_query(question, search_query):
        return None

    target = _extract_file_lookup_target(question)
    terms = _build_file_lookup_terms(question, search_query, target)
    if not terms and not target:
        return None

    file_records: dict[str, dict] = {}
    for idx in relevant_indices:
        try:
            path = repo_state.chunk_paths[idx]
            text = repo_state.chunk_texts[idx] or ""
        except Exception:
            continue
        rec = file_records.setdefault(path, {"texts": []})
        if text and len(rec["texts"]) < 4:
            rec["texts"].append(text)

    if not file_records:
        return None

    norm_by_file: dict[str, tuple[str, str]] = {}
    df_by_term: dict[str, int] = {}
    term_norm_cache: dict[str, str] = {}

    for path, rec in file_records.items():
        merged_text = "\n".join(rec["texts"])
        text_norm = _normalize_lookup_token(merged_text)
        path_norm = _normalize_lookup_token(path)
        norm_by_file[path] = (text_norm, path_norm)

    for term in terms:
        t_norm = _normalize_lookup_token(term)
        if not t_norm:
            continue
        term_norm_cache[term] = t_norm
        hit_count = 0
        for text_norm, path_norm in norm_by_file.values():
            if t_norm in text_norm or t_norm in path_norm:
                hit_count += 1
        if hit_count > 0:
            df_by_term[t_norm] = hit_count

    target_norm = _normalize_lookup_token(target)
    exact_target_hit_files: set[str] = set()
    scored: list[tuple[str, float, list[str]]] = []

    for path, rec in file_records.items():
        text_norm, path_norm = norm_by_file[path]
        score = 0.0
        matched_terms: list[str] = []

        if target_norm:
            if target_norm in text_norm:
                score += 4.2
                exact_target_hit_files.add(path)
                matched_terms.append(target)
            if target_norm in path_norm:
                score += 5.0
                exact_target_hit_files.add(path)
                if target not in matched_terms:
                    matched_terms.append(target)

        for term in terms:
            t_norm = term_norm_cache.get(term) or _normalize_lookup_token(term)
            if not t_norm:
                continue

            hit_text = t_norm in text_norm
            hit_path = t_norm in path_norm
            if not hit_text and not hit_path:
                continue

            df = max(1, df_by_term.get(t_norm, 1))
            rarity = 1.0 / float(df)
            base = 0.8 + min(len(t_norm), 10) * 0.16
            if hit_text:
                score += base + rarity
            if hit_path:
                score += base * 1.15 + rarity
            if term not in matched_terms:
                matched_terms.append(term)

        if score > 0:
            scored.append((path, score, matched_terms))

    if not scored:
        return None

    scored.sort(key=lambda x: (-x[1], x[0]))

    if logger:
        top_debug = [f"{path}({score:.2f})" for path, score, _ in scored[:5]]
        logger.info(f"   \U0001f9ea [\u6587\u4ef6\u5b9a\u4f4d\u672c\u5730\u6253\u5206] target={target or '<none>'} | top={top_debug}")

    if len(exact_target_hit_files) == 1:
        best_path = next(iter(exact_target_hit_files))
        display_target = target or "该内容"
        return f"{display_target}在文件【{best_path}】中。"

    best_path, best_score, _ = scored[0]
    if len(scored) == 1:
        display_target = target or "该内容"
        return f"{display_target}在文件【{best_path}】中。"

    second_score = scored[1][1]
    if best_score >= second_score * 1.35 and (best_score - second_score) >= 1.2:
        display_target = target or "该内容"
        return f"{display_target}在文件【{best_path}】中。"

    lines = ["根据当前检索结果，可能涉及以下文件："]
    for i, (path, _, _) in enumerate(scored[: max(2, max_items)], start=1):
        lines.append(f"{i}. {path}")
    return "\n".join(lines)


def extract_related_topic(question: str) -> str:
    q = re.sub(r"\s+", "", (question or "").strip())
    if not q:
        return ""

    patterns = [
        r"(?:最近)?有哪些?和(.+?)(?:有关|相关)(?:的)?(?:记录|文档|文件)",
        r"(.+?)(?:有关|相关)(?:的)?(?:记录|文档|文件)有哪些?",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if not m:
            continue
        topic = (m.group(1) or "").strip("，。！？；：,.!?;:")
        topic = re.sub(r"^(和|与|跟|关于)", "", topic)
        return topic
    return ""


def _build_topic_variants(topic: str) -> list[str]:
    t = (topic or "").strip().lower()
    if not t:
        return []

    variants: list[str] = []

    def _add(x: str):
        s = (x or "").strip().lower()
        if s and s not in variants:
            variants.append(s)

    _add(t)

    for token in re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fa5]{2,}", t):
        _add(token)

    if re.fullmatch(r"[\u4e00-\u9fa5]{4,}", t):
        _add(t[:2])
        _add(t[-2:])
        for i in range(len(t) - 1):
            _add(t[i:i + 2])

    return variants


def maybe_build_related_records_answer(question: str, relevant_indices, repo_state, max_items: int = 8) -> str | None:
    if not is_related_record_listing_request(question):
        return None

    topic = extract_related_topic(question)
    if not topic:
        return None

    variants = _build_topic_variants(topic)
    if not variants:
        return None

    file_hits: dict[str, dict] = {}

    for idx in relevant_indices or []:
        path = repo_state.chunk_paths[idx]
        text = repo_state.chunk_texts[idx] or ""
        path_lower = path.lower()
        text_lower = text.lower()

        score = 0
        for v in variants:
            if v in path_lower:
                score = max(score, 3)
            if v in text_lower:
                score = max(score, 2)

        if score <= 0:
            continue

        evidence = ""
        for line in text.splitlines():
            ln = (line or "").strip()
            if not ln:
                continue
            if any(v in ln.lower() for v in variants):
                evidence = ln
                break

        rec = file_hits.get(path)
        dt = repo_state.chunk_file_times[idx]
        if not rec or score > rec["score"]:
            file_hits[path] = {
                "path": path,
                "score": score,
                "dt": dt,
                "evidence": evidence,
            }

    if not file_hits:
        return None

    items = [x for x in file_hits.values() if x["score"] >= 2]
    if not items:
        items = sorted(file_hits.values(), key=lambda x: x["score"], reverse=True)[:3]

    items = sorted(items, key=lambda x: x["dt"], reverse=True)[:max_items]

    lines = [f"根据现有记录，和“{topic}”有关的记录有 {len(items)} 条："]
    for i, item in enumerate(items, 1):
        path = item["path"]
        date_match = re.search(r"(20\d{2}-\d{2}-\d{2})", Path(path).name)
        date_text = date_match.group(1) if date_match else item["dt"].strftime("%Y-%m-%d")
        lines.append(f"{i}. {path}（{date_text}）")
        if item["evidence"]:
            ev = item["evidence"]
            if len(ev) > 80:
                ev = ev[:80] + "..."
            lines.append(f"   证据：{ev}")

    return "\n".join(lines)
