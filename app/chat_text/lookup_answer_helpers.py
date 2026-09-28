from __future__ import annotations

import re

from app.chat_text.lookup_common import *
from app.chat_text.lookup_predicates import *
from app.chat_text.lookup_fact_candidates import (
    _match_quantity_attribute_propositions,
    _quantity_attribute_slots,
)


LOCAL_SUFFICIENCY_NO_ANSWER = "NO_ANSWER"
LOCAL_SUFFICIENCY_PARTIAL = "PARTIAL"
LOCAL_SUFFICIENCY_SUFFICIENT = "SUFFICIENT"


def _enumerated_quantity_targets(question: str) -> tuple[str, ...]:
    """Return explicit requested fields from a multi-value quantity question."""
    target = re.sub(r"^\s*(?:根据|按照|按|依据)[^，,。？?]+[，,]", "", question or "")
    ending = re.search(r"(?:分别|各自)?(?:是|为)?多少[？?]?\s*$", target)
    if not ending:
        return ()
    target = target[:ending.start()].strip()
    if "的" in target:
        target = target.split("的", 1)[1]
    labels = tuple(
        norm
        for label in re.split(r"、|以及|和|[，,]", target)
        if (norm := _normalize_lookup_token(label))
    )
    return labels if len(labels) >= 2 else ()


def _quantity_value_count(text: str) -> int:
    """Count distinct value expressions without assigning domain semantics."""
    value_pattern = re.compile(
        r"\d{4}年\d{1,2}月\d{1,2}日"
        r"|(?:>=|<=|≥|≤|>|<)?\s*\d+(?:\.\d+)?\s*"
        r"(?:至|～|~|-|–|—)\s*\d+(?:\.\d+)?\s*"
        r"(?:[a-zA-Z%‰℃°]+|[\u4e00-\u9fa5]{1,4})?"
        r"|(?:>=|<=|≥|≤|>|<)?\s*\d+(?:\.\d+)?\s*"
        r"(?:[a-zA-Z%‰℃°]+|[\u4e00-\u9fa5]{1,4})"
    )
    count = 0
    for match in value_pattern.finditer(text or ""):
        # A source edition is a qualifier, not one of the requested values.
        if (text or "")[match.start():match.end() + 1].endswith("年版"):
            continue
        count += 1
    return count


def assess_direct_lookup_sufficiency(question: str, items: list[dict]) -> str:
    """Decide whether selected direct evidence may own the final answer.

    Single-fact and non-quantity direct lookups keep their existing local-first
    behavior.  An explicit multi-value request is final only when the selected
    evidence covers every named field, or an already-admitted proposition has
    at least one requested-field binding and enough distinct values to answer
    the complete list.  Otherwise the existing generation path must take over.
    """
    if not items:
        return LOCAL_SUFFICIENCY_NO_ANSWER
    targets = _enumerated_quantity_targets(question)
    if not targets:
        return LOCAL_SUFFICIENCY_SUFFICIENT

    evidence_norm = _normalize_lookup_token("\n".join(str(item.get("line") or "") for item in items))
    covered = {target for target in targets if target in evidence_norm}
    matched_terms = {
        _normalize_lookup_token(str(term))
        for item in items
        for term in (item.get("matched_terms") or ())
        if _normalize_lookup_token(str(term))
    }
    covered.update(target for target in targets if target in matched_terms)
    if len(covered) == len(targets):
        return LOCAL_SUFFICIENCY_SUFFICIENT

    value_count = sum(_quantity_value_count(str(item.get("line") or "")) for item in items)
    if covered and value_count >= len(targets):
        return LOCAL_SUFFICIENCY_SUFFICIENT
    return LOCAL_SUFFICIENCY_PARTIAL

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


def _is_quantity_lookup_question(question: str) -> bool:
    return bool(re.search(r"多少[？?]?\s*$", question or ""))


def _direct_lookup_fact_terms(question: str, terms: list[str]) -> list[str] | None:
    """Separate a factual lookup's subject from its source reference."""
    target = re.sub(r"^\s*(?:根据|按照|按|依据)[^，,。？?]+[，,]", "", question or "")
    if not (_is_quantity_lookup_question(target) or re.search(r"是否|有没有|有无|能否|可否|需不需要", target)):
        return None
    target_norm = _normalize_lookup_token(target)
    return [
        term for term in terms
        if (norm := _normalize_lookup_token(term)) in target_norm
        and not norm.isdigit()
        and norm not in DIRECT_LOOKUP_STOP_TERMS
        and norm not in {"根据", "按照", "依据", "是否", "有没有", "有无", "能否", "可否", "需不需要"}
    ]


def _is_factual_lookup_statement(
    line: str, matched_terms: list[str], *, require_quantity: bool = False,
) -> bool:
    # A topical heading is not a factual answer. Keep the complete source
    # paragraph so that its negation and dependent conditions stay together.
    if not re.search(r"[。.!！；;][\"'”’）)]?$", line):
        return False
    body = re.sub(r"^\s*[（(]?[一二三四五六七八九十\d]+[)）.、]\s*", "", line)
    if require_quantity and not re.search(r"\d", body):
        return False
    remainder = _normalize_lookup_token(body)
    for term in sorted(matched_terms, key=len, reverse=True):
        remainder = remainder.replace(term, "")
    return len(remainder) >= 3


def _build_direct_lookup_evidence_items(
    *,
    terms: list[str],
    focus_terms: list[str],
    relevant_indices,
    repo_state,
    max_items: int,
    question: str = "",
) -> list[dict]:
    fact_terms = _direct_lookup_fact_terms(question, terms)
    factual_lookup = fact_terms is not None
    quantity_lookup = _is_quantity_lookup_question(question)
    attribute_topic, attribute_slots = _quantity_attribute_slots(question) if quantity_lookup else ("", [])
    if factual_lookup:
        terms = fact_terms
        focus_terms = fact_terms
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
    factual_coverage: dict[tuple[str, str], frozenset[str]] = {}

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
            if not factual_lookup and len(raw_line) > 160:
                continue

            raw_line_lower = raw_line.lower()
            line_norm = _normalize_lookup_token(raw_line)
            if not line_norm:
                continue

            hits = 0
            score = 0.0
            matched_focus = False
            matched_terms: list[str] = []
            for t in term_norms:
                if _term_matches_line(t, raw_line_lower, line_norm):
                    matched_terms.append(t)
                    hits += 1
                    weight = weighted_terms.get(t, 1.0)
                    score += (0.8 + min(len(t), 10) * 0.08) * weight
                    if t in focus_norms:
                        matched_focus = True
            nonliteral_matches = {}
            if hits == 0 and attribute_slots:
                nonliteral_matches = _match_quantity_attribute_propositions(
                    raw_line, text, attribute_topic, attribute_slots,
                )
                for label in nonliteral_matches:
                    matched_terms.append(label)
                    hits += 1
                    score += (0.8 + min(len(label), 10) * 0.08) * 2.0
                    matched_focus = True
            if hits <= 0:
                continue
            if factual_lookup and not _is_factual_lookup_statement(
                raw_line, matched_terms, require_quantity=quantity_lookup,
            ):
                continue
            if factual_lookup:
                factual_coverage[(str(path), raw_line)] = frozenset(matched_terms)

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
                    "matched_terms": tuple(matched_terms),
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
                    "matched_terms": item["matched_terms"],
                }
            )

    ranked.sort(key=lambda x: (-float(x["score"]), x["path"], x["line"]))
    if factual_lookup:
        # Source diversity must not reintroduce a paragraph that only covers
        # a strict subset of the question terms covered by stronger evidence.
        coverages = [factual_coverage[(x["path"], x["line"])] for x in ranked]
        ranked = [
            item for item, coverage in zip(ranked, coverages)
            if not any(coverage < other for other in coverages)
        ]

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
