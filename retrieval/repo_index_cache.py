from __future__ import annotations

from pathlib import Path

import numpy as np

from retrieval.repo_index_types import CacheSnapshot, ManifestDiff, ReuseResult


def _safe_load_object(cache, key: str, default):
    if key not in cache.files:
        return default
    arr = cache[key]
    try:
        return arr.item()
    except Exception:
        try:
            return arr[0]
        except Exception:
            return default



def load_cache_snapshot(cache_file: Path, logger) -> CacheSnapshot:
    if not cache_file.exists():
        return CacheSnapshot(manifest={}, doc_cache={}, chunk_cache={}, usable=False)

    try:
        cache = np.load(cache_file, allow_pickle=True)
        manifest = _safe_load_object(cache, "manifest", {})
        doc_cache = _safe_load_object(cache, "doc_cache", {})
        chunk_cache = _safe_load_object(cache, "chunk_cache", {})

        if isinstance(manifest, dict) and isinstance(doc_cache, dict) and isinstance(chunk_cache, dict):
            logger.info("✨ 检测到增量缓存，将执行差量比对")
            return CacheSnapshot(manifest=manifest, doc_cache=doc_cache, chunk_cache=chunk_cache, usable=True)

        logger.info("⚠️ 缓存结构不是增量版，将重建一次并升级缓存格式。")
    except Exception as e:
        logger.warning(f"⚠️ 读取缓存失败，将重建缓存：{e}")

    return CacheSnapshot(manifest={}, doc_cache={}, chunk_cache={}, usable=False)



def classify_manifest_diff(current_paths: list[str], current_manifest: dict[str, str], snapshot: CacheSnapshot) -> ManifestDiff:
    added_paths: list[str] = []
    modified_paths: list[str] = []
    unchanged_paths: list[str] = []

    if snapshot.usable:
        for path in current_paths:
            new_fp = current_manifest[path]
            old_fp = snapshot.manifest.get(path)
            if old_fp is None:
                added_paths.append(path)
            elif old_fp != new_fp:
                modified_paths.append(path)
            else:
                unchanged_paths.append(path)
    else:
        added_paths = list(current_paths)

    deleted_paths: list[str] = []
    if snapshot.usable:
        for old_path in snapshot.manifest.keys():
            if old_path not in current_manifest:
                deleted_paths.append(old_path)

    return ManifestDiff(
        added_paths=added_paths,
        modified_paths=modified_paths,
        unchanged_paths=unchanged_paths,
        deleted_paths=deleted_paths,
    )



def reuse_unchanged_entries(
    unchanged_paths: list[str],
    old_doc_cache: dict[str, dict],
    old_chunk_cache: dict[str, dict],
    path_to_fp: dict[str, str],
) -> ReuseResult:
    new_doc_cache: dict[str, dict] = {}
    new_chunk_cache: dict[str, dict] = {}
    promoted_modified_paths: list[str] = []
    reused_count = 0

    for path in unchanged_paths:
        doc_entry = old_doc_cache.get(path)
        chunk_entry = old_chunk_cache.get(path)

        if (
            isinstance(doc_entry, dict)
            and isinstance(chunk_entry, dict)
            and doc_entry.get("fingerprint") == path_to_fp[path]
            and chunk_entry.get("fingerprint") == path_to_fp[path]
        ):
            new_doc_cache[path] = doc_entry
            new_chunk_cache[path] = chunk_entry
            reused_count += 1
        else:
            promoted_modified_paths.append(path)

    return ReuseResult(
        new_doc_cache=new_doc_cache,
        new_chunk_cache=new_chunk_cache,
        reused_count=reused_count,
        promoted_modified_paths=promoted_modified_paths,
    )



def save_incremental_cache(
    cache_file: Path,
    current_manifest: dict[str, str],
    new_doc_cache: dict[str, dict],
    new_chunk_cache: dict[str, dict],
) -> None:
    np.savez(
        cache_file,
        manifest=np.array(current_manifest, dtype=object),
        doc_cache=np.array(new_doc_cache, dtype=object),
        chunk_cache=np.array(new_chunk_cache, dtype=object),
    )
