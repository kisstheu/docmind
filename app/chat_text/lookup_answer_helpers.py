from __future__ import annotations

import re

from app.chat_text.lookup_common import *
from app.chat_text.lookup_predicates import *

def _looks_like_direct_lookup_question(question: str) -> bool:
    q = _normalize_lookup_token(question)
    if not q:
        return False
    if any(marker in q for marker in DIRECT_LOOKUP_NON_LOOKUP_MARKERS):
        return False
    if any(marker in q for marker in DIRECT_LOOKUP_ANALYSIS_MARKERS):
        return False
    has_lookup_signal = any(marker in q for marker in DIRECT_LOOKUP_MARKERS)
    has_focus_signal = bool(re.search(r"[a-zA-Z0-9\u4e00-\u9fa5]{2,}", question or ""))
    return has_lookup_signal and has_focus_signal


def _looks_like_direct_lookup_followup_question(question: str) -> bool:
    q = _normalize_lookup_token(question)
    if not q:
        return False
    if _looks_like_direct_lookup_question(question):
        return True
    if any(marker in q for marker in DIRECT_LOOKUP_NON_LOOKUP_MARKERS):
        return False
    if any(marker in q for marker in DIRECT_LOOKUP_ANALYSIS_MARKERS):
        return False
    has_followup_signal = any(marker in q for marker in DIRECT_LOOKUP_FOLLOWUP_MARKERS)
    has_focus_signal = bool(re.search(r"[a-zA-Z0-9\u4e00-\u9fa5]{2,}", question or ""))
    has_selector_signal = bool(DIRECT_LOOKUP_SELECTOR_PATTERN.search(question or ""))
    return has_followup_signal and (has_focus_signal or has_selector_signal)


def _extract_direct_lookup_terms(question: str, search_query: str) -> list[str]:
    from retrieval.query_utils import extract_query_terms

    terms: list[str] = []
    seen: set[str] = set()

    for raw in extract_query_terms(search_query or "", question or ""):
        token = (raw or "").strip()
        norm = _normalize_lookup_token(token)
        if not norm:
            continue
        if norm in DIRECT_LOOKUP_STOP_TERMS:
            continue
        if len(norm) <= 1:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        terms.append(token)

    if terms:
        return terms

    raw_terms = re.findall(r"[a-zA-Z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", question or "")
    for token in raw_terms:
        norm = _normalize_lookup_token(token)
        if not norm or norm in DIRECT_LOOKUP_STOP_TERMS or norm in seen:
            continue
        seen.add(norm)
        terms.append(token)
    return terms


def _extract_direct_lookup_focus_terms(question: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    for token in re.findall(r"[a-zA-Z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", question or ""):
        norm = _normalize_lookup_token(token)
        if not norm:
            continue
        if norm in DIRECT_LOOKUP_STOP_TERMS:
            continue
        if len(norm) <= 1:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        terms.append(token)
    return terms


def _build_direct_lookup_evidence_items(
    *,
    terms: list[str],
    focus_terms: list[str],
    relevant_indices,
    repo_state,
    max_items: int,
) -> list[dict]:
    term_norms = [_normalize_lookup_token(t) for t in terms if _normalize_lookup_token(t)]
    if not term_norms:
        return []
    selector_query_hit = any(
        re.fullmatch(r"\d{1,4}", t) or bool(RANGE_SIGNATURE_PATTERN.search(t))
        for t in term_norms
    )

    focus_norms = {_normalize_lookup_token(t) for t in focus_terms if _normalize_lookup_token(t)}
    has_position_focus = any(_looks_like_position_term(norm) for norm in focus_norms)
    weighted_terms: dict[str, float] = {}
    for t in term_norms:
        if t in weighted_terms:
            continue
        weighted_terms[t] = 2.0 if t in focus_norms else 1.0

    ranked: list[dict] = []
    seen_lines: set[tuple[str, str]] = set()

    for idx in relevant_indices or []:
        try:
            path = repo_state.chunk_paths[idx]
            text = repo_state.chunk_texts[idx] or ""
        except Exception:
            continue

        per_path_items: list[dict] = []
        for line in text.splitlines():
            raw_line = (line or "").strip()
            if not raw_line:
                continue
            if len(raw_line) > 160:
                continue

            raw_line_lower = raw_line.lower()
            line_norm = _normalize_lookup_token(raw_line)
            if not line_norm:
                continue

            hits = 0
            score = 0.0
            matched_focus = False
            for t in term_norms:
                if _term_matches_line(t, raw_line_lower, line_norm):
                    hits += 1
                    weight = weighted_terms.get(t, 1.0)
                    score += (0.8 + min(len(t), 10) * 0.08) * weight
                    if t in focus_norms:
                        matched_focus = True
            if hits <= 0:
                continue

            if hits >= 2:
                score += 0.35
            if DIRECT_LOOKUP_STRUCTURED_LINE_PATTERN.search(raw_line):
                score += 0.18
            if 6 <= len(raw_line) <= 80:
                score += 0.12
            if RANGE_SIGNATURE_PATTERN.search(raw_line):
                score += 0.42
                if selector_query_hit:
                    score += 0.32
            if _looks_like_heading_line(raw_line):
                score += 0.28
            if _looks_like_detail_line(raw_line):
                score -= 0.18
            if has_position_focus:
                line_tokens = re.findall(r"[a-zA-Z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", raw_line)
                line_has_position = any(
                    _looks_like_position_term(_normalize_lookup_token(tok))
                    for tok in line_tokens
                )
                matched_position_term = any(
                    _looks_like_position_term(t) and _term_matches_line(t, raw_line_lower, line_norm)
                    for t in term_norms
                )
                if not line_has_position and not matched_position_term:
                    continue
                if not line_has_position and not _looks_like_heading_line(raw_line):
                    score -= 0.25

            if focus_norms and matched_focus:
                score += 0.45
            elif focus_norms and not matched_focus:
                score *= 0.62

            per_path_items.append(
                {
                    "path": str(path),
                    "line": raw_line,
                    "line_norm": line_norm,
                    "score": score,
                    "matched_focus": matched_focus,
                }
            )

        if not per_path_items:
            continue

        per_path_items.sort(key=lambda x: (-float(x["score"]), x["line"]))
        for item in per_path_items[:3]:
            dedup_key = (item["path"], item["line_norm"])
            if dedup_key in seen_lines:
                continue
            seen_lines.add(dedup_key)
            ranked.append(
                {
                    "path": item["path"],
                    "line": item["line"],
                    "score": item["score"],
                    "matched_focus": item["matched_focus"],
                }
            )

    ranked.sort(key=lambda x: (-float(x["score"]), x["path"], x["line"]))

    def _select_diverse(items: list[dict]) -> list[dict]:
        selected: list[dict] = []
        used_paths: set[str] = set()

        for item in items:
            path = str(item.get("path") or "")
            if not path or path in used_paths:
                continue
            used_paths.add(path)
            selected.append(item)
            if len(selected) >= max_items:
                return selected

        return selected

    focus_ranked = [x for x in ranked if x.get("matched_focus")]
    if focus_norms and focus_ranked:
        return _select_diverse(focus_ranked)
    return _select_diverse(ranked)

