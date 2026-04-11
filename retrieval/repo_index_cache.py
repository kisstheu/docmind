from __future__ import annotations

from pathlib import Path

import numpy as np

from retrieval.repo_index_types import ArchivedReuseResult, CacheSnapshot, ManifestDiff, ReuseResult

_REQUIRED_SCENE_TAG_VERSION = 2


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
        return CacheSnapshot(
            manifest={},
            doc_cache={},
            chunk_cache={},
            archived_manifest={},
            archived_doc_cache={},
            archived_chunk_cache={},
            usable=False,
        )

    try:
        cache = np.load(cache_file, allow_pickle=True)
        manifest = _safe_load_object(cache, "manifest", {})
        doc_cache = _safe_load_object(cache, "doc_cache", {})
        chunk_cache = _safe_load_object(cache, "chunk_cache", {})
        archived_manifest = _safe_load_object(cache, "archived_manifest", {})
        archived_doc_cache = _safe_load_object(cache, "archived_doc_cache", {})
        archived_chunk_cache = _safe_load_object(cache, "archived_chunk_cache", {})

        if (
            isinstance(manifest, dict)
            and isinstance(doc_cache, dict)
            and isinstance(chunk_cache, dict)
            and isinstance(archived_manifest, dict)
            and isinstance(archived_doc_cache, dict)
            and isinstance(archived_chunk_cache, dict)
        ):
            logger.info("✨ 检测到增量缓存，将执行差量比对")
            return CacheSnapshot(
                manifest=manifest,
                doc_cache=doc_cache,
                chunk_cache=chunk_cache,
                archived_manifest=archived_manifest,
                archived_doc_cache=archived_doc_cache,
                archived_chunk_cache=archived_chunk_cache,
                usable=True,
            )

        logger.info("⚠️ 缓存结构不是增量版，将重建一次并升级缓存格式。")
    except Exception as exc:
        logger.warning(f"⚠️ 读取缓存失败，将重建缓存：{exc}")

    return CacheSnapshot(
        manifest={},
        doc_cache={},
        chunk_cache={},
        archived_manifest={},
        archived_doc_cache={},
        archived_chunk_cache={},
        usable=False,
    )


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


def _is_doc_entry_reusable(doc_entry: dict, expected_fingerprint: str) -> bool:
    if not isinstance(doc_entry, dict):
        return False
    if doc_entry.get("fingerprint") != expected_fingerprint:
        return False
    if "scene_tags" not in doc_entry:
        return False
    if not isinstance(doc_entry.get("scene_tags", ""), str):
        return False
    return int(doc_entry.get("scene_tags_version", 0) or 0) >= _REQUIRED_SCENE_TAG_VERSION


def _reuse_cached_entry(
    path: str,
    expected_fingerprint: str,
    doc_cache: dict[str, dict],
    chunk_cache: dict[str, dict],
) -> tuple[dict, dict] | None:
    doc_entry = doc_cache.get(path)
    chunk_entry = chunk_cache.get(path)
    if (
        _is_doc_entry_reusable(doc_entry, expected_fingerprint)
        and isinstance(chunk_entry, dict)
        and chunk_entry.get("fingerprint") == expected_fingerprint
    ):
        return doc_entry, chunk_entry
    return None


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
        cached_entry = _reuse_cached_entry(path, path_to_fp[path], old_doc_cache, old_chunk_cache)
        if cached_entry is None:
            promoted_modified_paths.append(path)
            continue

        doc_entry, chunk_entry = cached_entry
        new_doc_cache[path] = doc_entry
        new_chunk_cache[path] = chunk_entry
        reused_count += 1

    return ReuseResult(
        new_doc_cache=new_doc_cache,
        new_chunk_cache=new_chunk_cache,
        reused_count=reused_count,
        promoted_modified_paths=promoted_modified_paths,
    )


def reuse_added_entries_from_archive(
    added_paths: list[str],
    archived_manifest: dict[str, str],
    archived_doc_cache: dict[str, dict],
    archived_chunk_cache: dict[str, dict],
    path_to_fp: dict[str, str],
) -> ArchivedReuseResult:
    new_doc_cache: dict[str, dict] = {}
    new_chunk_cache: dict[str, dict] = {}
    remaining_added_paths: list[str] = []
    reused_count = 0

    for path in added_paths:
        expected_fingerprint = path_to_fp[path]
        if archived_manifest.get(path) != expected_fingerprint:
            remaining_added_paths.append(path)
            continue

        cached_entry = _reuse_cached_entry(path, expected_fingerprint, archived_doc_cache, archived_chunk_cache)
        if cached_entry is None:
            remaining_added_paths.append(path)
            continue

        doc_entry, chunk_entry = cached_entry
        new_doc_cache[path] = doc_entry
        new_chunk_cache[path] = chunk_entry
        reused_count += 1

    return ArchivedReuseResult(
        new_doc_cache=new_doc_cache,
        new_chunk_cache=new_chunk_cache,
        reused_count=reused_count,
        remaining_added_paths=remaining_added_paths,
    )


def save_incremental_cache(
    cache_file: Path,
    current_manifest: dict[str, str],
    new_doc_cache: dict[str, dict],
    new_chunk_cache: dict[str, dict],
    archived_manifest: dict[str, str],
    archived_doc_cache: dict[str, dict],
    archived_chunk_cache: dict[str, dict],
) -> None:
    np.savez(
        cache_file,
        manifest=np.array(current_manifest, dtype=object),
        doc_cache=np.array(new_doc_cache, dtype=object),
        chunk_cache=np.array(new_chunk_cache, dtype=object),
        archived_manifest=np.array(archived_manifest, dtype=object),
        archived_doc_cache=np.array(archived_doc_cache, dtype=object),
        archived_chunk_cache=np.array(archived_chunk_cache, dtype=object),
    )
