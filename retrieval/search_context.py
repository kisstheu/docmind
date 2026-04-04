from __future__ import annotations

from typing import List

from retrieval.chunking import expand_neighbor_chunks
from retrieval.query_utils import classify_org_candidate, extract_company_candidates


def build_context_text(relevant_indices: List[int], repo_state, logger) -> str:
    if not relevant_indices:
        return ""

    expanded_indices = expand_neighbor_chunks(
        top_chunk_indices=relevant_indices,
        chunk_paths=repo_state.chunk_paths,
        chunk_meta=repo_state.chunk_meta,
        neighbor=1,
    )

    file_chunk_count = {}
    filtered_indices = []
    for idx in expanded_indices:
        p = repo_state.chunk_paths[idx]
        file_chunk_count.setdefault(p, 0)
        if file_chunk_count[p] >= 3:
            continue
        filtered_indices.append(idx)
        file_chunk_count[p] += 1

    context_blocks = []
    for idx in filtered_indices:
        meta = repo_state.chunk_meta[idx]
        context_blocks.append(
            f"文件【{repo_state.chunk_paths[idx]}】（chunk #{meta['chunk_id']}，位置 {meta['start']}-{meta['end']}）：\n{repo_state.chunk_texts[idx]}"
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

