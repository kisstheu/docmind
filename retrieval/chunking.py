from __future__ import annotations

from typing import Any, Dict, List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    chunks = []
    start = 0
    text = text.strip()
    if not text:
        return chunks
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]
        chunks.append({"text": chunk, "start": start, "end": end})
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _neighbor_indices(idx, chunk_paths, chunk_meta, neighbor: int):
    current_path = chunk_paths[idx]
    current_chunk_id = chunk_meta[idx]["chunk_id"]
    for candidate_idx in range(
        max(0, idx - neighbor),
        min(len(chunk_paths), idx + neighbor + 1),
    ):
        if (
            chunk_paths[candidate_idx] == current_path
            and abs(chunk_meta[candidate_idx]["chunk_id"] - current_chunk_id) <= neighbor
        ):
            yield candidate_idx


def expand_neighbor_chunks(top_chunk_indices, chunk_paths, chunk_meta, neighbor: int = 1):
    expanded = set()
    for idx in top_chunk_indices:
        expanded.update(_neighbor_indices(idx, chunk_paths, chunk_meta, neighbor))
    return sorted(expanded)


def select_seed_and_neighbor_chunks(
    seed_indices,
    chunk_paths,
    chunk_meta,
    *,
    neighbor: int = 1,
    per_file_limit: int = 3,
):
    if per_file_limit <= 0:
        return []

    unique_seed_indices = []
    seen_seed_indices = set()
    for idx in seed_indices:
        if idx in seen_seed_indices:
            continue
        seen_seed_indices.add(idx)
        unique_seed_indices.append(idx)

    selected_indices = set()
    selected_seed_indices = []
    file_chunk_counts = {}

    # seed_indices 保留初始相关性顺序；先让 seed 占用每文件预算。
    for idx in unique_seed_indices:
        path = chunk_paths[idx]
        if file_chunk_counts.get(path, 0) >= per_file_limit:
            continue
        selected_indices.add(idx)
        selected_seed_indices.append(idx)
        file_chunk_counts[path] = file_chunk_counts.get(path, 0) + 1

    # 邻块只服务于已保留 seed，并按 seed 排名、距离和局部位置确定补充顺序。
    neighbor_priority = {}
    for seed_rank, seed_idx in enumerate(selected_seed_indices):
        seed_chunk_id = chunk_meta[seed_idx]["chunk_id"]
        for candidate_idx in _neighbor_indices(
            seed_idx,
            chunk_paths,
            chunk_meta,
            neighbor,
        ):
            if candidate_idx in seen_seed_indices:
                continue
            priority = (
                seed_rank,
                abs(chunk_meta[candidate_idx]["chunk_id"] - seed_chunk_id),
                chunk_meta[candidate_idx]["chunk_id"],
                candidate_idx,
            )
            previous_priority = neighbor_priority.get(candidate_idx)
            if previous_priority is None or priority < previous_priority:
                neighbor_priority[candidate_idx] = priority

    for candidate_idx, _priority in sorted(
        neighbor_priority.items(),
        key=lambda item: item[1],
    ):
        path = chunk_paths[candidate_idx]
        if file_chunk_counts.get(path, 0) >= per_file_limit:
            continue
        selected_indices.add(candidate_idx)
        file_chunk_counts[path] = file_chunk_counts.get(path, 0) + 1

    # 文档位置仅在选择完成后决定展示顺序。
    path_order = {}
    for idx, path in enumerate(chunk_paths):
        path_order.setdefault(path, idx)
    return sorted(
        selected_indices,
        key=lambda idx: (
            path_order[chunk_paths[idx]],
            chunk_meta[idx]["chunk_id"],
            idx,
        ),
    )
