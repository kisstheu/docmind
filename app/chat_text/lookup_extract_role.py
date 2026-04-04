from __future__ import annotations

import re

from app.chat_text.lookup_common import *
from app.chat_text.lookup_predicates import *

def _extract_role_name_items(
    *,
    role_terms: list[str],
    anchor_terms: list[str],
    required_selector_signatures: set[str] | None = None,
    relevant_indices,
    repo_state,
    max_items: int,
) -> list[dict]:
    role_norms = [_normalize_lookup_token(t) for t in role_terms if _normalize_lookup_token(t)]
    if not role_norms:
        return []

    anchor_norms = [_normalize_lookup_token(t) for t in anchor_terms if _normalize_lookup_token(t)]
    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()
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
        text_lower = "\n".join(lines).lower()
        text_norm = _normalize_lookup_token("\n".join(lines))
        if required_selector_signatures:
            file_selector_sigs = _extract_selector_signatures(text_lower)
            if not _selector_constraints_satisfied(file_selector_sigs, set(required_selector_signatures)):
                continue
        file_anchor_hits = 0
        if anchor_norms:
            for term in anchor_norms:
                if _term_matches_text(term, text_lower, text_norm):
                    file_anchor_hits += 1
            min_file_anchor_hits = 2 if len(anchor_norms) >= 3 else 1
            if file_anchor_hits < min_file_anchor_hits:
                continue

        for i, line in enumerate(lines):
            line_norm = _normalize_lookup_token(line)
            if not line_norm:
                continue
            if not any(_term_matches_line(t, line.lower(), line_norm) for t in role_norms):
                continue

            for wi in range(i - 4, i + 5):
                if wi < 0 or wi >= len(lines):
                    continue
                evidence = lines[wi]
                if not evidence or len(evidence) > 180:
                    continue

                for pattern in PERSON_NAME_PATTERNS:
                    for m in pattern.findall(evidence):
                        name = (m or "").strip()
                        if not name:
                            continue
                        if not _is_plausible_person_name(name, evidence):
                            continue
                        key = (path, name)
                        if key in seen:
                            continue
                        seen.add(key)

                        score = 1.0
                        if wi == i:
                            score += 0.6
                        if "·" in evidence or ":" in evidence or "：" in evidence:
                            score += 0.2
                        if re.search(r"(先生|女士|老师)$", name):
                            score += 0.4
                        if file_anchor_hits:
                            score += min(file_anchor_hits, 4) * 0.3
                        elif anchor_norms:
                            score *= 0.4
                        company = _extract_company_from_nearby_lines(lines, i, radius=5)
                        if company:
                            score += 0.25

                        candidates.append(
                            {
                                "name": name,
                                "path": path,
                                "evidence": evidence,
                                "company": company,
                                "score": score,
                            }
                        )

    candidates.sort(key=lambda x: (-float(x["score"]), x["path"], x["name"]))
    if anchor_norms and candidates:
        best_score = float(candidates[0]["score"])
        candidates = [x for x in candidates if float(x["score"]) >= best_score - 0.25]
    return candidates[:max_items]

