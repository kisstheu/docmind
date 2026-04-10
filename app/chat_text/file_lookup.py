from __future__ import annotations

import re

from app.chat_text.lookup_answer_main import (
    FILE_LOOKUP_FOLLOWUP_PRONOUNS,
    FILE_LOOKUP_GENERIC_TERMS,
    FILE_LOOKUP_POLITE_PREFIXES,
)

_ANALYTIC_FILE_LOOKUP_BLOCK_PATTERNS = (
    re.compile(r"(?:\u8981\u6c42|\u6280\u80fd|\u638c\u63e1|\u6280\u672f\u6808|\u80fd\u529b|\u7ecf\u9a8c).*(?:\u54ea\u4e9b|\u4ec0\u4e48|\u76f8\u5bf9\u8f83\u4f4e|\u5207\u5165\u53e3)"),
    re.compile(r"\u54ea.*(?:\u6280\u672f|\u6280\u80fd)"),
    re.compile(r"\u66f4\u5bb9\u6613.*\u5207\u5165\u53e3"),
)

def _normalize_lookup_token(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "", (text or "").lower())


def _looks_like_analytic_followup_question(question: str) -> bool:
    q = re.sub(r"\s+", "", (question or ""))
    if not q:
        return False
    if any(word in q for word in ("\u6587\u4ef6", "\u6587\u6863", "\u8bb0\u5f55")) and any(
        marker in q for marker in ("\u54ea\u4e2a", "\u54ea\u4efd", "\u54ea\u7bc7", "\u5728\u54ea")
    ):
        return False
    return any(pattern.search(q) for pattern in _ANALYTIC_FILE_LOOKUP_BLOCK_PATTERNS)


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


def _extract_followup_file_lookup_target(question: str) -> str:
    q = _strip_file_lookup_prefix(question)
    if not q:
        return ""

    q = re.sub(r"\s+", "", q)
    q = q.strip("，,：:。！？!?`\"'[]【】")
    q = re.sub(r"(?:呢|吗|嘛|呀|啊)+$", "", q)
    q = re.sub(r"(?:怎么样|如何|咋样|怎么说|怎么理解)$", "", q)
    q = q.strip("，,：:。！？!?`\"'[]【】")
    if not q:
        return ""

    q_norm = _normalize_lookup_token(q)
    if not q_norm:
        return ""
    if q_norm in FILE_LOOKUP_GENERIC_TERMS:
        return ""
    if q_norm in FILE_LOOKUP_FOLLOWUP_PRONOUNS:
        return ""
    if re.fullmatch(r"\d+(?:\.\d+)?", q_norm):
        return ""
    if len(q_norm) <= 1:
        return ""
    if not re.search(r"[a-zA-Z\u4e00-\u9fa5]", q):
        return ""
    return q


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
    allow_followup_inference: bool = False,
) -> str | None:
    # Local import to avoid any potential cross-module initialization coupling.
    from retrieval.search_intent import is_file_location_lookup_query

    if not relevant_indices:
        return None

    is_direct_lookup = is_file_location_lookup_query(question, search_query)
    target = _extract_file_lookup_target(question)
    if not is_direct_lookup:
        if _looks_like_analytic_followup_question(question):
            return None
        if not allow_followup_inference:
            return None
        if not target:
            target = _extract_followup_file_lookup_target(question)
        if not target:
            return None
        if logger:
            logger.info(f"   🧷 [文件定位追问推断] target={target}")

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
