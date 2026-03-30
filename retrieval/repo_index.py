from __future__ import annotations

from pathlib import Path

from retrieval.repo_index_build import (
    assemble_repo_state,
    build_changed_file_cache_entry,
    clean_shadow_tags,
    collect_all_files,
    read_changed_file,
    scan_repository as _scan_repository,
)
from retrieval.repo_index_cache import (
    _safe_load_object,
    classify_manifest_diff,
    load_cache_snapshot,
    reuse_unchanged_entries,
    save_incremental_cache,
)
from retrieval.repo_index_types import IndexBuildContext, RepoState


__all__ = [
    "RepoState",
    "collect_all_files",
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



def load_or_build_embeddings(scanned, cache_file: Path, model_emb, logger, ollama_api_url: str, ollama_model: str):
    scanned_repo = _scan_repository(scanned["notes_dir"], logger) if isinstance(scanned, dict) else scanned

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

    new_doc_cache = reuse_result.new_doc_cache
    new_chunk_cache = reuse_result.new_chunk_cache
    final_changed_paths = diff.added_paths + diff.modified_paths + reuse_result.promoted_modified_paths

    if reuse_result.reused_count:
        logger.info(f"♻️ 已复用 {reuse_result.reused_count} 个未变化文件的索引结果")

    if final_changed_paths:
        logger.info("\n🧠 检测到新增或修改文件，开始增量建库...\n")

    context = IndexBuildContext(
        notes_dir=scanned_repo.notes_dir,
        model_emb=model_emb,
        logger=logger,
        ollama_api_url=ollama_api_url,
        ollama_model=ollama_model,
    )

    sidecar_count = 0
    sidecar_examples: list[str] = []

    for path in dict.fromkeys(final_changed_paths):
        file_record = read_changed_file(scanned_repo.notes_dir, path, logger)
        if not file_record:
            logger.warning(f"      ⚠️ 跳过空文件或读取失败文件：{path}")
            continue

        if file_record.used_sidecar:
            sidecar_count += 1
            if len(sidecar_examples) < 5:
                sidecar_examples.append(path + ".ocr.txt")

        cache_pair = build_changed_file_cache_entry(
            context,
            path,
            path_to_fp[path],
            file_record=file_record,
        )
        if cache_pair is None:
            continue
        doc_entry, chunk_entry = cache_pair
        new_doc_cache[path] = doc_entry
        new_chunk_cache[path] = chunk_entry

    if sidecar_count > 0:
        logger.info(f"      📄 本轮变动文件中共命中 {sidecar_count} 个伴生文件")
        if sidecar_examples:
            logger.info(f"         例如: {', '.join(sidecar_examples)}")

    repo_state = assemble_repo_state(scanned_repo, current_paths, new_doc_cache, new_chunk_cache)
    save_incremental_cache(cache_file, current_manifest, new_doc_cache, new_chunk_cache)

    if final_changed_paths or diff.deleted_paths:
        logger.info("✅ 增量索引更新完成")
    else:
        logger.info("✅ 所有索引均直接复用，无需重建")

    return repo_state
