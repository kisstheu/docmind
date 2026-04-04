from __future__ import annotations

import re

from app.chat_text.lookup_common import *
from app.chat_text.lookup_predicates import *

def _extract_company_hr_mapping_items(
    *,
    role_terms: list[str],
    anchor_terms: list[str],
    required_selector_signatures: set[str],
    relevant_indices,
    repo_state,
    max_items: int,
) -> list[dict]:
    role_norms = [_normalize_lookup_token(t) for t in role_terms if _normalize_lookup_token(t)]
    specific_role_norms = [t for t in role_norms if t not in GENERIC_ROLE_NORMS and len(t) >= 2]
    anchor_norms = [_normalize_lookup_token(t) for t in anchor_terms if _normalize_lookup_token(t)]
    require_role_match = bool(role_norms)

    items: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    processed_paths: set[str] = set()
    doc_text_lookup = _build_doc_text_lookup(repo_state)

    for idx in relevant_indices or []:
        try:
            path = str(repo_state.chunk_paths[idx])
            if path in processed_paths:
                continue
            processed_paths.add(path)
            chunk_text = repo_state.chunk_texts[idx] or ""
            text = doc_text_lookup.get(path) or chunk_text
        except Exception:
            continue

        lines = [ln.strip() for ln in text.splitlines() if (ln or "").strip()]
        if not lines:
            continue

        file_text_lower = "\n".join(lines).lower()
        file_text_norm = _normalize_lookup_token("\n".join(lines))
        file_anchor_hits = 0
        for term in anchor_norms:
            if _term_matches_text(term, file_text_lower, file_text_norm):
                file_anchor_hits += 1
        min_file_anchor_hits = 2 if len(anchor_norms) >= 3 else 1
        if anchor_norms and file_anchor_hits < min_file_anchor_hits:
            continue
        if required_selector_signatures:
            file_selector_sigs = _extract_selector_signatures(file_text_lower)
            if not _selector_constraints_satisfied(file_selector_sigs, required_selector_signatures):
                continue

        company_candidates: list[tuple[int, str]] = []
        person_candidates: list[tuple[int, str, str]] = []
        heading_candidates: list[tuple[int, str]] = []
        role_match_scores: dict[int, float] = {}
        specific_role_scores: dict[int, float] = {}
        for li, li_text in enumerate(lines):
            company = _extract_company_from_line(li_text)
            if not company and li + 1 < len(lines):
                company = _extract_company_from_split_lines(li_text, lines[li + 1])
            if company:
                company_candidates.append((li, company))
            if _looks_like_heading_line(li_text):
                heading_candidates.append((li, li_text))
            if require_role_match:
                role_score = _role_line_match_score(li_text, role_norms)
                target_role_score = (
                    _role_line_match_score(li_text, specific_role_norms)
                    if specific_role_norms
                    else role_score
                )
                if target_role_score >= 0.55:
                    role_match_scores[li] = role_score
                    specific_role_scores[li] = target_role_score
            for pattern in PERSON_NAME_PATTERNS:
                for m in pattern.findall(li_text):
                    name = (m or "").strip()
                    if name and _is_plausible_person_name(name, li_text):
                        person_candidates.append((li, name, li_text))

        if require_role_match and not role_match_scores:
            continue
        if not company_candidates or not person_candidates:
            continue

        file_item_count_before = len(items)
        role_score_terms = specific_role_norms or role_norms

        def _nearest_company(center: int, max_dist: int) -> tuple[str, int]:
            best_name = ""
            best_dist = 10_000
            for ci, cname in company_candidates:
                dist = abs(ci - center)
                if dist <= max_dist and dist < best_dist:
                    best_dist = dist
                    best_name = cname
            return best_name, best_dist

        def _nearest_person(center: int, max_dist: int) -> tuple[str, str, int]:
            best_name = ""
            best_line = ""
            best_dist = 10_000
            for pi, pname, pline in person_candidates:
                dist = abs(pi - center)
                if dist <= max_dist and dist < best_dist:
                    best_dist = dist
                    best_name = pname
                    best_line = pline
            return best_name, best_line, best_dist

        def _best_heading(center: int) -> str:
            best_line = ""
            best_dist = 10_000
            for hi, hline in heading_candidates:
                dist = abs(hi - center)
                if dist <= 12 and dist < best_dist:
                    best_dist = dist
                    best_line = hline
            if best_line:
                return best_line
            return lines[center]

        def _best_role_line(center: int) -> tuple[str, float]:
            best_line = _best_heading(center)
            best_rank = _role_line_match_score(best_line, role_score_terms) if role_score_terms else 0.0
            start = max(0, center - 10)
            end = min(len(lines), center + 11)
            for li in range(start, end):
                li_text = lines[li]
                role_score = _role_line_match_score(li_text, role_score_terms) if role_score_terms else 0.0
                if require_role_match and role_score <= 0.0:
                    continue
                rank = role_score
                if _looks_like_heading_line(li_text):
                    rank += 0.22
                if _looks_like_detail_line(li_text):
                    rank -= 0.18
                rank -= max(0, abs(li - center) - 1) * 0.08
                if rank > best_rank:
                    best_rank = rank
                    best_line = li_text
            return best_line, max(0.0, _role_line_match_score(best_line, role_score_terms) if role_score_terms else 0.0)

        if require_role_match:
            candidate_centers = sorted(role_match_scores.keys(), key=lambda i: (-role_match_scores[i], i))
        elif anchor_norms:
            candidate_centers = [
                i
                for i, line in enumerate(lines)
                if _normalize_lookup_token(line)
                and any(_term_matches_line(t, line.lower(), _normalize_lookup_token(line)) for t in anchor_norms)
            ]
        else:
            candidate_centers = list(range(len(lines)))

        for i in candidate_centers:
            line = lines[i]
            line_norm = _normalize_lookup_token(line)
            if not line_norm:
                continue

            role_match_score = role_match_scores.get(i, 0.0)
            if require_role_match and role_match_score <= 0.0:
                continue
            role_specific_score = specific_role_scores.get(i, role_match_score)
            if require_role_match and specific_role_norms and role_specific_score <= 0.0:
                continue

            line_has_anchor = any(_term_matches_line(t, line.lower(), line_norm) for t in anchor_norms) if anchor_norms else False
            if not require_role_match and anchor_norms and not line_has_anchor:
                continue

            local_lines = lines[max(0, i - 20): min(len(lines), i + 21)]
            local_lower = "\n".join(local_lines).lower()
            local_norm = _normalize_lookup_token("\n".join(local_lines))
            local_anchor_hits = 0
            for term in anchor_norms:
                if _term_matches_text(term, local_lower, local_norm):
                    local_anchor_hits += 1

            company, company_dist = _nearest_company(i, 8)
            used_company_fallback = False
            if not company:
                company, company_dist = _nearest_company(i, 60)
                used_company_fallback = bool(company)
            if not company:
                continue

            nearest_name, nearest_name_line, nearest_dist = _nearest_person(i, 12)
            used_name_fallback = False
            if not nearest_name:
                nearest_name, nearest_name_line, nearest_dist = _nearest_person(i, 80)
                used_name_fallback = bool(nearest_name)
            if not nearest_name:
                continue

            role_line, role_line_match_score = _best_role_line(i)
            if anchor_norms:
                role_line_norm = _normalize_lookup_token(role_line)
                if not any(_term_matches_line(t, role_line.lower(), role_line_norm) for t in anchor_norms):
                    best_anchor_line = ""
                    best_anchor_rank = -1.0
                    for li in local_lines:
                        li_norm = _normalize_lookup_token(li)
                        if not li_norm:
                            continue
                        if not any(_term_matches_line(t, li.lower(), li_norm) for t in anchor_norms):
                            continue
                        li_role_score = _role_line_match_score(li, role_score_terms) if role_score_terms else 0.0
                        if require_role_match and li_role_score < 0.2:
                            continue
                        li_rank = li_role_score + (0.2 if _looks_like_heading_line(li) else 0.0)
                        if li_rank > best_anchor_rank:
                            best_anchor_rank = li_rank
                            best_anchor_line = li
                    if best_anchor_line:
                        role_line = best_anchor_line
                        role_line_match_score = _role_line_match_score(role_line, role_score_terms) if role_score_terms else 0.0

            score = 1.0
            score += min(file_anchor_hits, 6) * 0.24
            score += min(local_anchor_hits, 5) * 0.40
            score += max(0, 1.1 - nearest_dist * 0.14)
            score += max(0, 0.7 - company_dist * 0.09)
            score += min(role_specific_score, 1.8) * 0.75
            score += min(role_line_match_score, 1.6) * 0.55
            if used_company_fallback:
                score -= 0.20
            if used_name_fallback:
                score -= 0.20
            if role_line and len(role_line) <= 100:
                score += 0.15
            if _looks_like_detail_line(role_line):
                score -= 0.25
            if require_role_match and role_line_match_score < 0.45:
                score -= 0.45

            key = (path, company, nearest_name)
            if key in seen:
                continue
            seen.add(key)

            items.append(
                {
                    "path": path,
                    "company": company,
                    "name": nearest_name,
                    "role_line": role_line,
                    "evidence": nearest_name_line,
                    "score": score,
                }
            )

        if (not require_role_match) and len(items) == file_item_count_before:
            # Fallback: allow file-level pairing when OCR layout breaks local adjacency.
            pair = None
            best_dist = 10_000
            for ci, cname in company_candidates:
                for pi, pname, pline in person_candidates:
                    dist = abs(ci - pi)
                    if dist < best_dist:
                        best_dist = dist
                        pair = (cname, pname, pline, pi)
            if pair is not None:
                company, pname, pline, pidx = pair
                role_line = _best_heading(pidx)
                local_lines = lines[max(0, pidx - 20): min(len(lines), pidx + 21)]
                local_lower = "\n".join(local_lines).lower()
                local_norm = _normalize_lookup_token("\n".join(local_lines))
                local_anchor_hits = 0
                for term in anchor_norms:
                    if _term_matches_text(term, local_lower, local_norm):
                        local_anchor_hits += 1
                key = (path, company, pname)
                if key not in seen:
                    seen.add(key)
                    score = 0.9 + min(file_anchor_hits, 6) * 0.22 + min(local_anchor_hits, 4) * 0.3
                    score += max(0, 0.6 - best_dist * 0.06)
                    items.append(
                        {
                            "path": path,
                            "company": company,
                            "name": pname,
                            "role_line": role_line,
                            "evidence": pline,
                            "score": score,
                        }
                    )

    items.sort(key=lambda x: (-float(x["score"]), x["path"], x["company"], x["name"]))
    selected: list[dict] = []
    used_paths: set[str] = set()
    cap = min(max_items, 4)
    for item in items:
        path = str(item.get("path") or "")
        if not path:
            continue
        if path in used_paths:
            continue
        used_paths.add(path)
        selected.append(item)
        if len(selected) >= cap:
            break
    return selected

