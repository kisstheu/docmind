from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from retrieval.repo_index_build import (
    assemble_repo_state,
    clean_scene_tags,
    clean_shadow_tags,
    collect_all_files,
    populate_prepared_file_tags,
    prepare_changed_file_for_indexing,
    read_changed_file,
    scan_repository as _scan_repository,
)
from retrieval.repo_index_cache import (
    _safe_load_object,
    classify_manifest_diff,
    load_cache_snapshot,
    reuse_added_entries_from_archive,
    reuse_unchanged_entries,
    save_incremental_cache,
)
from retrieval.repo_index_encode import build_cache_entries_from_prepared
from retrieval.repo_index_types import IndexBuildContext, RepoState


def _coerce_float(raw, default: float, min_value: float, max_value: float) -> float:
    try:
        value = float(raw)
    except Exception:
        return default
    return max(min_value, min(max_value, value))


def _coerce_int(raw, default: int, min_value: int, max_value: int) -> int:
    try:
        value = int(raw)
    except Exception:
        return default
    return max(min_value, min(max_value, value))


def _coerce_str(raw, default: str) -> str:
    value = str(raw or "").strip()
    return value or default


def _default_prepare_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(4, cpu_count))


def _scale_ollama_timeout(timeout_sec: float, tag_concurrency: int) -> float:
    concurrency = max(1, int(tag_concurrency or 1))
    if concurrency <= 1:
        return timeout_sec
    return timeout_sec * (1.0 + 0.5 * (concurrency - 1))


def _build_archived_cache(snapshot, current_paths: list[str], deleted_paths: list[str]) -> tuple[dict[str, str], dict[str, dict], dict[str, dict]]:
    archived_manifest = dict(snapshot.archived_manifest)
    archived_doc_cache = dict(snapshot.archived_doc_cache)
    archived_chunk_cache = dict(snapshot.archived_chunk_cache)

    for path in deleted_paths:
        old_fingerprint = snapshot.manifest.get(path)
        old_doc_entry = snapshot.doc_cache.get(path)
        old_chunk_entry = snapshot.chunk_cache.get(path)
        if old_fingerprint is None or old_doc_entry is None or old_chunk_entry is None:
            continue
        archived_manifest[path] = old_fingerprint
        archived_doc_cache[path] = old_doc_entry
        archived_chunk_cache[path] = old_chunk_entry

    for path in current_paths:
        archived_manifest.pop(path, None)
        archived_doc_cache.pop(path, None)
        archived_chunk_cache.pop(path, None)

    return archived_manifest, archived_doc_cache, archived_chunk_cache


__all__ = [
    "RepoState",
    "collect_all_files",
    "clean_scene_tags",
    "clean_shadow_tags",
    "read_changed_file",
    "scan_repository",
    "load_or_build_embeddings",
]



def scan_repository(notes_dir: Path, logger):
    return _scan_repository(notes_dir, logger).to_legacy_dict()



def _coerce_scanned_repo(scanned):
    if hasattr(scanned, "entries") and hasattr(scanned, "notes_dir"):
        return scanned
    return _scan_repository(scanned["notes_dir"], logger=None) if False else None



def load_or_build_embeddings(
    scanned,
    cache_file: Path,
    model_emb,
    logger,
    ollama_api_url: str,
    ollama_model: str,
    ollama_timeout_sec: float | None = None,
    ollama_max_retries: int | None = None,
):
    scanned_repo = _scan_repository(scanned["notes_dir"], logger) if isinstance(scanned, dict) else scanned
    resolved_ollama_timeout = _coerce_float(
        ollama_timeout_sec if ollama_timeout_sec is not None else os.getenv("DOCMIND_INDEX_OLLAMA_TIMEOUT_SEC"),
        default=8.0,
        min_value=2.0,
        max_value=180.0,
    )
    resolved_ollama_retries = _coerce_int(
        ollama_max_retries if ollama_max_retries is not None else os.getenv("DOCMIND_INDEX_OLLAMA_RETRIES"),
        default=0,
        min_value=0,
        max_value=5,
    )
    resolved_prepare_workers = _coerce_int(
        os.getenv("DOCMIND_INDEX_PREPARE_WORKERS"),
        default=_default_prepare_workers(),
        min_value=1,
        max_value=8,
    )
    resolved_embed_batch_size = _coerce_int(
        os.getenv("DOCMIND_INDEX_EMBED_BATCH_SIZE"),
        default=16,
        min_value=1,
        max_value=128,
    )
    resolved_tag_batch_size = _coerce_int(
        os.getenv("DOCMIND_INDEX_TAG_BATCH_SIZE"),
        default=1,
        min_value=1,
        max_value=12,
    )
    resolved_tag_ollama_model = _coerce_str(
        os.getenv("DOCMIND_INDEX_TAG_OLLAMA_MODEL"),
        default=ollama_model,
    )
    resolved_tag_ollama_keep_alive = _coerce_str(
        os.getenv("DOCMIND_INDEX_TAG_OLLAMA_KEEP_ALIVE"),
        default="30m",
    )
    resolved_tag_num_predict = _coerce_int(
        os.getenv("DOCMIND_INDEX_TAG_OLLAMA_NUM_PREDICT"),
        default=32,
        min_value=8,
        max_value=128,
    )
    resolved_tag_num_ctx = _coerce_int(
        os.getenv("DOCMIND_INDEX_TAG_OLLAMA_NUM_CTX"),
        default=2048,
        min_value=256,
        max_value=8192,
    )
    resolved_tag_temperature = _coerce_float(
        os.getenv("DOCMIND_INDEX_TAG_OLLAMA_TEMPERATURE"),
        default=0.0,
        min_value=0.0,
        max_value=1.0,
    )
    resolved_tag_mode = (os.getenv("DOCMIND_INDEX_TAG_MODE") or "ollama").strip().lower()
    if resolved_tag_mode not in {"statistical", "ollama"}:
        resolved_tag_mode = "ollama"
    resolved_tag_concurrency = _coerce_int(
        os.getenv("DOCMIND_INDEX_TAG_CONCURRENCY"),
        default=2 if resolved_tag_mode == "ollama" else 1,
        min_value=1,
        max_value=4,
    )
    effective_ollama_timeout = _scale_ollama_timeout(
        resolved_ollama_timeout,
        resolved_tag_concurrency if resolved_tag_mode == "ollama" else 1,
    )

    current_paths = scanned_repo.paths
    current_manifest = {entry.path: entry.fingerprint for entry in scanned_repo.entries}
    path_to_fp = current_manifest.copy()

    snapshot = load_cache_snapshot(cache_file, logger)
    diff = classify_manifest_diff(current_paths, current_manifest, snapshot)

    logger.info(
        f"📊 本轮索引差量统计：新增 {len(diff.added_paths)} / 修改 {len(diff.modified_paths)} / 删除 {len(diff.deleted_paths)} / 复用 {len(diff.unchanged_paths)}"
    )

    reuse_result = reuse_unchanged_entries(
        unchanged_paths=diff.unchanged_paths,
        old_doc_cache=snapshot.doc_cache,
        old_chunk_cache=snapshot.chunk_cache,
        path_to_fp=path_to_fp,
    )
    archived_reuse_result = reuse_added_entries_from_archive(
        added_paths=diff.added_paths,
        archived_manifest=snapshot.archived_manifest,
        archived_doc_cache=snapshot.archived_doc_cache,
        archived_chunk_cache=snapshot.archived_chunk_cache,
        path_to_fp=path_to_fp,
    )

    new_doc_cache = reuse_result.new_doc_cache | archived_reuse_result.new_doc_cache
    new_chunk_cache = reuse_result.new_chunk_cache | archived_reuse_result.new_chunk_cache
    final_changed_paths = (
        archived_reuse_result.remaining_added_paths
        + diff.modified_paths
        + reuse_result.promoted_modified_paths
    )

    if reuse_result.reused_count:
        logger.info(f"♻️ 已复用 {reuse_result.reused_count} 个未变化文件的索引结果")
    if archived_reuse_result.reused_count:
        logger.info(f"♻️ 已复用 {archived_reuse_result.reused_count} 个暂离扫描后重新出现文件的索引结果")

    if final_changed_paths:
        logger.info("\n🧠 检测到新增或修改文件，开始增量建库...\n")

    context = IndexBuildContext(
        notes_dir=scanned_repo.notes_dir,
        model_emb=model_emb,
        logger=logger,
        ollama_api_url=ollama_api_url,
        ollama_model=resolved_tag_ollama_model,
        tag_mode=resolved_tag_mode,
        tag_concurrency=resolved_tag_concurrency,
        ollama_timeout_sec=effective_ollama_timeout,
        ollama_max_retries=resolved_ollama_retries,
        ollama_keep_alive=resolved_tag_ollama_keep_alive,
        ollama_request_options={
            "temperature": resolved_tag_temperature,
            "num_predict": resolved_tag_num_predict,
            "num_ctx": resolved_tag_num_ctx,
            "top_k": 20,
            "top_p": 0.9,
        },
    )
    if resolved_tag_mode == "ollama":
        logger.info(
            f"⚙️ 建库标签提取: ollama (model={resolved_tag_ollama_model}, base_timeout={resolved_ollama_timeout:.1f}s, effective_timeout={effective_ollama_timeout:.1f}s, retries={resolved_ollama_retries}, batch_size={resolved_tag_batch_size}, concurrency={resolved_tag_concurrency}, keep_alive={resolved_tag_ollama_keep_alive}, num_predict={resolved_tag_num_predict}, num_ctx={resolved_tag_num_ctx})"
        )
    else:
        logger.info("⚙️ 建库标签提取: statistical")

    sidecar_count = 0
    sidecar_examples: list[str] = []
    unique_changed_paths = list(dict.fromkeys(final_changed_paths))
    total_changed = len(unique_changed_paths)
    changed_start = time.time()
    prepare_workers = min(resolved_prepare_workers, total_changed) if total_changed else 1

    if total_changed > 0:
        logger.info(
            f"   [build] prepare_workers={prepare_workers}, embed_batch_size={resolved_embed_batch_size}, tag_mode={resolved_tag_mode}, tag_batch_size={resolved_tag_batch_size}, tag_concurrency={resolved_tag_concurrency}"
        )
        logger.info(f"   📌 预处理进度 [0/{total_changed}] 0%")

    prepared_by_path: dict[str, object] = {}
    future_to_path: dict[object, str] = {}
    if total_changed > 0:
        with ThreadPoolExecutor(max_workers=prepare_workers, thread_name_prefix="index-build") as executor:
            for path in unique_changed_paths:
                future = executor.submit(
                    prepare_changed_file_for_indexing,
                    context,
                    path,
                    path_to_fp[path],
                )
                future_to_path[future] = path

            completed_prepare = 0
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    prepared = future.result()
                except Exception as exc:
                    completed_prepare += 1
                    progress_pct = int((completed_prepare / total_changed) * 100) if total_changed else 100
                    logger.warning(f"      ⚠️ {path} 预处理异常，已跳过 ({exc})")
                    logger.info(f"   📌 预处理进度 [{completed_prepare}/{total_changed}] {progress_pct}% -> {path}")
                    continue
                if prepared is None:
                    completed_prepare += 1
                    progress_pct = int((completed_prepare / total_changed) * 100) if total_changed else 100
                    logger.info(f"   📌 预处理进度 [{completed_prepare}/{total_changed}] {progress_pct}% -> {path}")
                    continue

                prepared_by_path[path] = prepared
                if prepared.file_record.used_sidecar:
                    sidecar_count += 1
                    if len(sidecar_examples) < 5:
                        sidecar_examples.append(path + ".ocr.txt")
                completed_prepare += 1
                progress_pct = int((completed_prepare / total_changed) * 100) if total_changed else 100
                logger.info(f"   📌 预处理进度 [{completed_prepare}/{total_changed}] {progress_pct}% -> {path}")

    ordered_prepared = [prepared_by_path[path] for path in unique_changed_paths if path in prepared_by_path]
    if ordered_prepared:
        populate_prepared_file_tags(
            context,
            ordered_prepared,
            tag_batch_size=resolved_tag_batch_size,
            tag_concurrency=resolved_tag_concurrency,
            existing_docs=[entry.get("doc", "") for entry in new_doc_cache.values()],
        )
        cache_entries = build_cache_entries_from_prepared(
            context,
            ordered_prepared,
            embed_batch_size=resolved_embed_batch_size,
        )
        for path in unique_changed_paths:
            cache_pair = cache_entries.get(path)
            if cache_pair is None:
                continue
            doc_entry, chunk_entry = cache_pair
            new_doc_cache[path] = doc_entry
            new_chunk_cache[path] = chunk_entry

    if total_changed > 0:
        logger.info(f"⏱️ 增量建库处理耗时: {time.time() - changed_start:.2f}s（共 {total_changed} 个文件）")

    if sidecar_count > 0:
        logger.info(f"      📄 本轮变动文件中共命中 {sidecar_count} 个伴生文件")
        if sidecar_examples:
            logger.info(f"         例如: {', '.join(sidecar_examples)}")

    archived_manifest, archived_doc_cache, archived_chunk_cache = _build_archived_cache(
        snapshot,
        current_paths=current_paths,
        deleted_paths=diff.deleted_paths,
    )

    repo_state = assemble_repo_state(scanned_repo, current_paths, new_doc_cache, new_chunk_cache)
    save_incremental_cache(
        cache_file,
        current_manifest,
        new_doc_cache,
        new_chunk_cache,
        archived_manifest,
        archived_doc_cache,
        archived_chunk_cache,
    )

    if final_changed_paths or diff.deleted_paths:
        logger.info("✅ 增量索引更新完成")
    else:
        logger.info("✅ 所有索引均直接复用，无需重建")

    return repo_state
