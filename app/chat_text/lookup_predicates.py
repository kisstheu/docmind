from __future__ import annotations

import re

from app.chat_text.lookup_common import *

def _is_role_like_term(term_norm: str) -> bool:
    t = (term_norm or "").strip().lower()
    if not t:
        return False
    if t in ROLE_TERM_HINTS:
        return True
    if re.fullmatch(r"[a-z]{2,12}", t):
        return t in ROLE_TERM_HINTS or _looks_like_position_term(t)
    if _looks_like_position_term(t):
        return True
    if any(
        t.endswith(suffix)
        for suffix in ("负责人", "联系人", "经理", "主管", "总监", "专员", "顾问", "老师", "秘书", "主任", "委员", "代表")
    ):
        return True
    return False


def _expand_role_terms(role_terms: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        t = (token or "").strip()
        n = _normalize_lookup_token(t)
        if not n or n in seen:
            return
        seen.add(n)
        expanded.append(t)

    for token in role_terms or []:
        _add(token)

    return expanded


def _extract_company_from_line(line: str) -> str:
    raw = (line or "").strip()
    if not raw:
        return ""

    for pattern in COMPANY_LINE_PATTERNS:
        m = pattern.search(raw)
        if not m:
            continue
        company = _sanitize_company_candidate((m.group(1) or ""), raw_line=raw)
        if not company:
            continue
        return company
    return ""


def _sanitize_company_candidate(company_raw: str, *, raw_line: str = "") -> str:
    company = (company_raw or "").strip(" ·:：[]【】()（）")
    if len(company) < 2:
        return ""
    if any(bad in company for bad in COMPANY_BAN_TERMS):
        return ""
    if _looks_like_address_fragment(company):
        return ""
    if raw_line and re.search(r"(省|市|区|县).*\d+(?:层|楼|室|号|栋)", raw_line):
        return ""
    if re.search(r"[，,。；;：:、\s]{2,}", company):
        return ""
    if company.startswith(("对", "将", "能", "熟练", "负责", "推动", "提升")):
        return ""
    return company


def _extract_company_from_split_lines(current_line: str, next_line: str) -> str:
    cur = (current_line or "").strip()
    nxt = (next_line or "").strip()
    if not cur or not nxt:
        return ""
    if not re.search(r"[·•]$", cur):
        return ""
    if not re.search(
        r"^(?:hrbp|hr|recruiter|talent|招聘(?:经理|专员|顾问|官)?|猎头|人才顾问)\b",
        nxt,
        flags=re.IGNORECASE,
    ):
        return ""
    base = re.sub(r"[·•]+$", "", cur).strip()
    return _sanitize_company_candidate(base, raw_line=f"{cur} {nxt}")


def _looks_like_address_fragment(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    # 地址常见特征：行政区串 + 门牌楼层/房间号等
    if re.search(r"(省|市|区|县|路|街|号|层|室|栋|单元)", raw) and re.search(r"\d", raw):
        return True
    if re.search(r"(?:[^\s]{2,8}(?:省|市|区)){2,}", raw):
        return True
    if re.search(r"(省|市|区|县|路|街|大道|巷|弄)", raw) and re.search(r"(中心|大厦|园区|广场|楼)", raw):
        return True
    return False


def _extract_company_from_nearby_lines(lines: list[str], center: int, radius: int = 4) -> str:
    if not lines:
        return ""
    start = max(0, center - radius)
    end = min(len(lines), center + radius + 1)
    for i in range(start, end):
        company = _extract_company_from_line(lines[i])
        if company:
            return company
        if i + 1 < len(lines):
            company = _extract_company_from_split_lines(lines[i], lines[i + 1])
            if company:
                return company
    return ""


def _build_doc_text_lookup(repo_state) -> dict[str, str]:
    paths = list(getattr(repo_state, "paths", []) or [])
    docs = list(getattr(repo_state, "docs", []) or [])
    if not paths or not docs:
        return {}

    max_count = min(len(paths), len(docs))
    out: dict[str, str] = {}
    for i in range(max_count):
        path = str(paths[i] or "")
        text = str(docs[i] or "")
        if not path or not text:
            continue
        prev = out.get(path, "")
        if len(text) > len(prev):
            out[path] = text
    return out


def _normalized_bigrams(text: str) -> set[str]:
    src = (text or "").strip()
    if len(src) < 2:
        return set()
    return {src[i : i + 2] for i in range(len(src) - 1)}


def _role_line_match_score(raw_line: str, role_norms: list[str]) -> float:
    line_raw = (raw_line or "").strip()
    if not line_raw:
        return 0.0
    line_lower = line_raw.lower()
    line_norm = _normalize_lookup_token(line_raw)
    if not line_norm:
        return 0.0

    line_bigrams = _normalized_bigrams(line_norm)
    best = 0.0
    for role in role_norms:
        role_norm = _normalize_lookup_token(role)
        if not role_norm:
            continue

        if _term_matches_line(role_norm, line_lower, line_norm):
            exact_score = 1.0 + min(len(role_norm), 20) * 0.02
            if exact_score > best:
                best = exact_score
            continue

        if len(role_norm) < 4:
            continue
        role_bigrams = _normalized_bigrams(role_norm)
        if not role_bigrams or not line_bigrams:
            continue

        overlap_hits = len(role_bigrams & line_bigrams)
        if overlap_hits < 2:
            continue
        ratio = overlap_hits / max(len(role_bigrams), 1)
        if ratio < 0.45:
            continue
        fuzzy_score = 0.45 + ratio * 0.65
        if fuzzy_score > best:
            best = fuzzy_score

    return best


def _looks_like_company_hr_mapping_query(question: str, focus_terms: list[str]) -> tuple[bool, list[str]]:
    q = (question or "").strip()
    q_norm = _normalize_lookup_token(q)
    if not q_norm:
        return False, []

    role_terms: list[str] = []
    for token in focus_terms:
        norm = _normalize_lookup_token(token)
        if norm and _is_role_like_term(norm) and token not in role_terms:
            role_terms.append(token)
    if not role_terms:
        return False, []

    has_relation_marker = any(marker in q_norm for marker in DIRECT_LOOKUP_MARKERS)
    has_org_marker = any(marker in q for marker in COMPANY_QUERY_MARKERS)
    if not has_relation_marker:
        return False, []
    if not has_org_marker and not any(x in q for x in ("对应", "关联", "匹配", "分别")):
        return False, []

    return True, role_terms


def _looks_like_position_term(term_norm: str) -> bool:
    t = (term_norm or "").strip().lower()
    if not t:
        return False
    if re.fullmatch(r"[a-z]{4,32}", t):
        if len(t) >= 7 and re.search(r"(?:er|or)$", t):
            return True
        if re.search(r"(?:ist|ian|ant|ive|ary|eer|ician|ologist|ographer)$", t):
            return True
    return any(
        t.endswith(suffix)
        for suffix in (
            "工程师",
            "架构师",
            "分析师",
            "研究员",
            "开发",
            "测试",
            "算法",
            "经理",
            "主管",
            "总监",
            "顾问",
            "职位",
            "岗位",
        )
    )


def _looks_like_mapping_followup_query(
    question: str,
    focus_terms: list[str],
    *,
    allow_followup_inference: bool,
) -> bool:
    if not allow_followup_inference:
        return False

    q = (question or "").strip()
    q_norm = _normalize_lookup_token(q)
    if not q_norm:
        return False

    if any(term in q_norm for term in MAPPING_FOLLOWUP_BLOCK_TERMS):
        return False

    if any(marker in q for marker in ROLE_NAME_QUERY_MARKERS):
        return False

    has_followup_signal = any(marker in q_norm for marker in DIRECT_LOOKUP_FOLLOWUP_MARKERS)
    has_selector_signal = bool(DIRECT_LOOKUP_SELECTOR_PATTERN.search(q)) or bool(RANGE_SIGNATURE_PATTERN.search(q))
    has_relation_signal = (
        any(marker in q for marker in COMPANY_QUERY_MARKERS)
        or any(marker in q_norm for marker in ("对应", "关联", "匹配", "映射"))
        or ("hr" in q_norm)
    )

    focus_norms = {_normalize_lookup_token(t) for t in focus_terms if _normalize_lookup_token(t)}
    has_anchor_focus = any(
        norm
        and norm not in DIRECT_LOOKUP_GENERIC_ANCHOR_TERMS
        and not _is_role_like_term(norm)
        for norm in focus_norms
    )
    has_position_focus = any(_looks_like_position_term(norm) for norm in focus_norms)

    if not (has_followup_signal or has_selector_signal):
        return False
    if not has_relation_signal:
        return False
    return has_anchor_focus or has_position_focus


def _looks_like_role_name_query(question: str, focus_terms: list[str]) -> tuple[bool, list[str]]:
    q = (question or "").strip()
    q_norm = _normalize_lookup_token(q)
    if not q_norm:
        return False, []

    role_terms: list[str] = []
    for token in focus_terms:
        norm = _normalize_lookup_token(token)
        if norm and _is_role_like_term(norm) and token not in role_terms:
            role_terms.append(token)
    if not role_terms:
        return False, []

    if any(marker in q for marker in ROLE_NON_NAME_QUERY_MARKERS):
        return False, []

    focus_norms = {_normalize_lookup_token(t) for t in focus_terms if _normalize_lookup_token(t)}
    role_norms = [_normalize_lookup_token(t) for t in role_terms if _normalize_lookup_token(t)]
    target_role_norms = [n for n in role_norms if not _looks_like_position_term(n)]
    has_anchor_focus = any(
        norm
        and norm not in DIRECT_LOOKUP_GENERIC_ANCHOR_TERMS
        and (not _is_role_like_term(norm) or _looks_like_position_term(norm))
        for norm in focus_norms
    )

    has_name_marker = any(marker in q for marker in ROLE_NAME_QUERY_MARKERS)
    has_short_followup_marker = len(q_norm) <= 10 and any(marker in q for marker in ROLE_NAME_FOLLOWUP_MARKERS)
    has_possessive_role_marker = (
        ("的" in q)
        and has_anchor_focus
        and bool(target_role_norms)
        and any(role_norm in q_norm for role_norm in target_role_norms)
    )
    if not has_name_marker and not has_short_followup_marker and not has_possessive_role_marker:
        return False, []

    return True, role_terms


__all__ = [name for name in globals() if not name.startswith("__")]
