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


def expand_neighbor_chunks(top_chunk_indices, chunk_paths, chunk_meta, neighbor: int = 1):
    expanded = set()
    for idx in top_chunk_indices:
        expanded.add(idx)
        current_path = chunk_paths[idx]
        current_chunk_id = chunk_meta[idx]["chunk_id"]
        for j in range(max(0, idx - neighbor), min(len(chunk_paths), idx + neighbor + 1)):
            if chunk_paths[j] == current_path and abs(chunk_meta[j]["chunk_id"] - current_chunk_id) <= neighbor:
                expanded.add(j)
    return sorted(expanded)
