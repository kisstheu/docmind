from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import List

from retrieval.chunking import (
    expand_neighbor_chunks,
    select_seed_and_neighbor_chunks,
)
from retrieval.query_utils import classify_org_candidate, extract_company_candidates


@dataclass(frozen=True)
class CanonicalSourceCandidate:
    """Immutable identity for one exact retrieval context chunk."""

    source_id: str
    repo_index: int
    path: str
    chunk_id: int
    start: int
    end: int
    text: str


def canonical_source_candidate_id(
    *,
    path: str,
    chunk_id: int,
    start: int,
    end: int,
    text: str,
) -> str:
    authority = json.dumps(
        {
            "path": path,
            "chunk_id": chunk_id,
            "start": start,
            "end": end,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(authority.encode("utf-8")).hexdigest()[:24]
    return f"retrieval-source:v1:{digest}"


def build_context_source_candidates(
    relevant_indices: List[int],
    repo_state,
    *,
    per_file_limit: int = 3,
    total_limit: int | None = None,
) -> tuple[CanonicalSourceCandidate, ...]:
    if not relevant_indices:
        return ()

    filtered_indices = select_seed_and_neighbor_chunks(
        seed_indices=relevant_indices,
        chunk_paths=repo_state.chunk_paths,
        chunk_meta=repo_state.chunk_meta,
        neighbor=1,
        per_file_limit=per_file_limit,
        total_limit=total_limit,
    )
    candidates: list[CanonicalSourceCandidate] = []
    for index in filtered_indices:
        meta = repo_state.chunk_meta[index]
        path = str(repo_state.chunk_paths[index])
        text = str(repo_state.chunk_texts[index])
        chunk_id = int(meta["chunk_id"])
        start = int(meta["start"])
        end = int(meta["end"])
        candidates.append(
            CanonicalSourceCandidate(
                source_id=canonical_source_candidate_id(
                    path=path,
                    chunk_id=chunk_id,
                    start=start,
                    end=end,
                    text=text,
                ),
                repo_index=index,
                path=path,
                chunk_id=chunk_id,
                start=start,
                end=end,
                text=text,
            )
        )
    return tuple(candidates)


def build_context_text(
    relevant_indices: List[int],
    repo_state,
    logger,
    *,
    per_file_limit: int = 3,
    total_limit: int | None = None,
    include_source_ids: bool = False,
) -> str:
    if not relevant_indices:
        return ""

    expanded_indices = expand_neighbor_chunks(
        top_chunk_indices=relevant_indices,
        chunk_paths=repo_state.chunk_paths,
        chunk_meta=repo_state.chunk_meta,
        neighbor=1,
    )
    source_candidates = build_context_source_candidates(
        relevant_indices,
        repo_state,
        per_file_limit=per_file_limit,
        total_limit=total_limit,
    )
    filtered_indices = [candidate.repo_index for candidate in source_candidates]

    def describe_indices(indices):
        return [
            (
                repo_state.chunk_paths[idx],
                repo_state.chunk_meta[idx]["chunk_id"],
            )
            for idx in indices
        ]

    seed_index_set = set(relevant_indices)
    logger.debug(
        "上下文候选角色: "
        f"seed={describe_indices(relevant_indices)} | "
        f"neighbor={describe_indices([idx for idx in expanded_indices if idx not in seed_index_set])}"
    )
    logger.debug(f"上下文每文件预算后: {describe_indices(filtered_indices)}")

    context_blocks = []
    for candidate in source_candidates:
        source_prefix = (
            f"证据【{candidate.source_id}】" if include_source_ids else ""
        )
        context_blocks.append(
            f"{source_prefix}文件【{candidate.path}】"
            f"（chunk #{candidate.chunk_id}，位置 {candidate.start}-{candidate.end}）：\n"
            f"{candidate.text}"
        )

    logger.debug(
        f"本轮检索命中的chunk文件列表: {list(dict.fromkeys([repo_state.chunk_paths[idx] for idx in filtered_indices]))}"
    )
    return "【参考片段】:\n" + "\n---\n".join(context_blocks) + "\n\n"


def uniq_keep_order(items):
    result = []
    for x in items:
        if x not in result:
            result.append(x)
    return result


def build_inventory_candidates_text(question: str, repo_state, inventory_target_type: str | None) -> str:
    if inventory_target_type != "company":
        return ""

    candidate_pool = []
    for doc_text in repo_state.docs:
        candidate_pool.extend(extract_company_candidates(doc_text))

    unique_names = []
    for name in candidate_pool:
        if name not in unique_names:
            unique_names.append(name)

    deduped_names = []
    for name in unique_names:
        if any(name != other and name in other and len(other) >= len(name) + 2 for other in unique_names):
            continue
        deduped_names.append(name)

    explicit_names, ambiguous_names, generic_names = [], [], []
    for name in deduped_names:
        kind = classify_org_candidate(name)
        if kind == "explicit":
            explicit_names.append(name)
        elif kind == "ambiguous":
            ambiguous_names.append(name)
        else:
            generic_names.append(name)

    explicit_names = uniq_keep_order(explicit_names)
    ambiguous_names = uniq_keep_order(ambiguous_names)
    generic_names = uniq_keep_order(generic_names)

    lines = []
    if explicit_names or ambiguous_names or generic_names:
        lines.append("【盘点候选：组织】")
        if explicit_names:
            lines.append("【明确组织名】")
            lines.extend([f"- {x}" for x in explicit_names[:20]])
        if ambiguous_names:
            lines.append("【可能是简称或未写全】")
            lines.extend([f"- {x}" for x in ambiguous_names[:20]])
        if generic_names:
            lines.append("【泛称/指代】")
            lines.extend([f"- {x}" for x in generic_names[:20]])

    return "\n".join(lines) + "\n\n" if lines else ""
