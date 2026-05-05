from __future__ import annotations

import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from loaders.file_loader import read_file
from retrieval.repo_index_encode import (
    _build_chunk_payloads,
    assemble_repo_state,
    build_cache_entries_from_prepared,
)
from retrieval.repo_index_scan import collect_all_files
from retrieval.repo_index_tags import (
    _SCENE_TAG_VERSION,
    _build_batch_tag_prompt,
    _build_combined_tag_prompt,
    _build_statistical_tag_stats,
    _extract_statistical_tags_for_indexing,
    _parse_batch_tag_response,
    _parse_combined_tag_response,
    clean_scene_tags,
    clean_shadow_tags,
)
from retrieval.repo_index_types import FileReadResult, IndexBuildContext, PreparedFileBuild, ScanEntry, ScannedRepo

_OLLAMA_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


def build_file_fingerprint(relative_path: str, stat_result) -> str:
    return f"{relative_path}|{stat_result.st_size}|{int(stat_result.st_mtime_ns)}"


def _format_file_info(relative_path: str, size_kb: float, mtime: datetime.datetime) -> str:
    return f"- {relative_path} (大小: {size_kb:.1f}KB, 更新于: {mtime.strftime('%Y-%m-%d')})"


def scan_repository(notes_dir: Path, logger) -> ScannedRepo:
    all_files = collect_all_files(notes_dir)
    entries: list[ScanEntry] = []
    file_info_list: list[str] = []

    for file in all_files:
        stat = file.stat()
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
        relative_path = file.relative_to(notes_dir).as_posix()
        entry = ScanEntry(
            path=relative_path,
            file_time=mtime,
            fingerprint=build_file_fingerprint(relative_path, stat),
            size_kb=stat.st_size / 1024,
        )
        entries.append(entry)
        file_info_list.append(_format_file_info(relative_path, entry.size_kb, mtime))

    return ScannedRepo(
        entries=entries,
        paths=[e.path for e in entries],
        file_times=[e.file_time for e in entries],
        file_info_list=file_info_list,
        all_files=all_files,
        earliest_note=file_info_list[0] if file_info_list else "无",
        latest_note=file_info_list[-1] if file_info_list else "无",
        notes_dir=notes_dir,
    )


def read_changed_file(notes_dir: Path, relative_path: str, logger) -> FileReadResult | None:
    file_path = notes_dir / relative_path
    stat = file_path.stat()
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
    content, used_sidecar = read_file(file_path, logger=logger)
    if not content:
        return None

    return FileReadResult(
        path=relative_path,
        doc=content,
        file_time=mtime,
        file_size=stat.st_size,
        file_info=_format_file_info(relative_path, stat.st_size / 1024, mtime),
        used_sidecar=used_sidecar,
    )


def _build_ollama_payload(context: IndexBuildContext, prompt: str) -> dict:
    payload = {
        "model": context.ollama_model,
        "prompt": prompt,
        "stream": False,
    }
    keep_alive = str(getattr(context, "ollama_keep_alive", "") or "").strip()
    if keep_alive:
        payload["keep_alive"] = keep_alive

    request_options = getattr(context, "ollama_request_options", None) or {}
    if request_options:
        payload["options"] = dict(request_options)
    return payload


def _request_ollama_response(context: IndexBuildContext, prompt: str, path: str, purpose: str) -> str:
    max_attempts = max(1, int(getattr(context, "ollama_max_retries", 0) or 0) + 1)
    timeout_sec = float(getattr(context, "ollama_timeout_sec", 8.0) or 8.0)
    payload = _build_ollama_payload(context, prompt)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.post(context.ollama_api_url, json=payload, timeout=timeout_sec)
            if response.status_code >= 400:
                error = requests.HTTPError(f"{response.status_code} {response.reason}", response=response)
                if response.status_code in _OLLAMA_RETRYABLE_STATUS_CODES and attempt < max_attempts:
                    context.logger.warning(
                        f"      ⚠️ {path} {purpose}失败，第 {attempt}/{max_attempts} 次重试中 (HTTP {response.status_code})"
                    )
                    continue
                raise error

            raw = str(response.json().get("response", "") or "").strip()
            return raw
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                context.logger.warning(f"      ⚠️ {path} {purpose}失败，第 {attempt}/{max_attempts} 次重试中 ({e})")
                continue
            break

    if last_error is None:
        raise RuntimeError(f"{purpose}失败，未获得可用响应")
    raise last_error


def _extract_tags_for_indexing(context: IndexBuildContext, doc: str, path: str) -> tuple[str, str]:
    context.logger.info(f"   🤖 正在透视文件：{path} ...")
    try:
        raw = _request_ollama_response(
            context=context,
            prompt=_build_combined_tag_prompt(doc),
            path=path,
            purpose="标签提取",
        )
        raw_shadow_tags, raw_scene_tags = _parse_combined_tag_response(raw)
        shadow_tags = clean_shadow_tags(raw_shadow_tags)
        scene_tags = clean_scene_tags(raw_scene_tags)

        if shadow_tags:
            context.logger.info(f"      ✅ 提取到影子标签：[{shadow_tags}]")
        if scene_tags:
            context.logger.info(f"      🧭 提取到场景标签：[{scene_tags}]")
        return shadow_tags, scene_tags
    except Exception as e:
        context.logger.warning(f"      ⚠️ {path} 透视失败，使用空标签 ({e})")
        return "", ""


def _process_ollama_tag_batch(
    context: IndexBuildContext,
    batch: list[PreparedFileBuild],
) -> None:
    if not batch:
        return

    if len(batch) == 1:
        prepared = batch[0]
        prepared.shadow_tags, prepared.scene_tags = _extract_tags_for_indexing(
            context,
            prepared.file_record.doc,
            prepared.path,
        )
        return

    path_summary = ", ".join(item.path for item in batch[:2])
    if len(batch) > 2:
        path_summary += ", ..."
    context.logger.info(f"   🤖 批量透视文件：{path_summary}")

    parsed_batch: dict[str, tuple[str, str]] = {}
    try:
        raw = _request_ollama_response(
            context=context,
            prompt=_build_batch_tag_prompt(batch),
            path=path_summary,
            purpose="批量标签提取",
        )
        parsed_batch = _parse_batch_tag_response(raw)
    except Exception as e:
        context.logger.warning(f"      ⚠️ 批量标签提取失败，回退单文件提取 ({e})")

    for idx, prepared in enumerate(batch, start=1):
        raw_shadow_tags, raw_scene_tags = parsed_batch.get(f"F{idx}", ("", ""))
        if raw_shadow_tags or raw_scene_tags:
            prepared.shadow_tags = clean_shadow_tags(raw_shadow_tags)
            prepared.scene_tags = clean_scene_tags(raw_scene_tags)
            if prepared.shadow_tags:
                context.logger.info(f"      ✅ 提取到影子标签：[{prepared.shadow_tags}]")
            if prepared.scene_tags:
                context.logger.info(f"      🧭 提取到场景标签：[{prepared.scene_tags}]")
            continue

        prepared.shadow_tags, prepared.scene_tags = _extract_tags_for_indexing(
            context,
            prepared.file_record.doc,
            prepared.path,
        )


def _summarize_tag_batch(batch: list[PreparedFileBuild]) -> str:
    if not batch:
        return ""
    if len(batch) == 1:
        return batch[0].path

    path_summary = ", ".join(item.path for item in batch[:2])
    if len(batch) > 2:
        path_summary += ", ..."
    return path_summary


def populate_prepared_file_tags(
    context: IndexBuildContext,
    prepared_files: list[PreparedFileBuild],
    tag_batch_size: int,
    existing_docs: list[str] | None = None,
    tag_concurrency: int = 1,
) -> None:
    if not prepared_files:
        return

    tag_mode = getattr(context, "tag_mode", "statistical")
    if tag_mode == "statistical":
        context.logger.info("   [build] tag_mode=statistical")
        stats = _build_statistical_tag_stats(existing_docs or [], prepared_files)
        total_files = len(prepared_files)
        context.logger.info(f"   🏷️ 标签进度 [0/{total_files}] 0%")
        for idx, prepared in enumerate(prepared_files, start=1):
            prepared.shadow_tags, prepared.scene_tags = _extract_statistical_tags_for_indexing(
                prepared.file_record.doc,
                stats,
            )
            progress_pct = int((idx / total_files) * 100) if total_files else 100
            context.logger.info(f"   🏷️ 标签进度 [{idx}/{total_files}] {progress_pct}% -> {prepared.path}")
        return

    batch_size = max(1, tag_batch_size)
    batches = [prepared_files[start:start + batch_size] for start in range(0, len(prepared_files), batch_size)]
    worker_count = min(max(1, tag_concurrency), len(batches))
    context.logger.info(f"   [build] tag_batch_size={batch_size}, tag_concurrency={worker_count}")
    total_files = len(prepared_files)
    context.logger.info(f"   🏷️ 标签进度 [0/{total_files}] 0%")

    if worker_count == 1:
        completed_files = 0
        for batch in batches:
            _process_ollama_tag_batch(context, batch)
            completed_files += len(batch)
            progress_pct = int((completed_files / total_files) * 100) if total_files else 100
            context.logger.info(
                f"   🏷️ 标签进度 [{completed_files}/{total_files}] {progress_pct}% -> {_summarize_tag_batch(batch)}"
            )
        return

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="index-tags") as executor:
        future_to_batch = {
            executor.submit(_process_ollama_tag_batch, context, batch): batch
            for batch in batches
        }
        completed_files = 0
        for future in as_completed(future_to_batch):
            future.result()
            batch = future_to_batch[future]
            completed_files += len(batch)
            progress_pct = int((completed_files / total_files) * 100) if total_files else 100
            context.logger.info(
                f"   🏷️ 标签进度 [{completed_files}/{total_files}] {progress_pct}% -> {_summarize_tag_batch(batch)}"
            )


def build_changed_file_cache_entry(
    context: IndexBuildContext,
    path: str,
    fingerprint: str,
    file_record: FileReadResult | None = None,
) -> tuple[dict, dict] | None:
    prepared = prepare_changed_file_for_indexing(
        context=context,
        path=path,
        fingerprint=fingerprint,
        file_record=file_record,
    )
    if prepared is None:
        return None

    populate_prepared_file_tags(context, [prepared], tag_batch_size=1)
    return build_cache_entries_from_prepared(context, [prepared], embed_batch_size=1).get(path)


def prepare_changed_file_for_indexing(
    context: IndexBuildContext,
    path: str,
    fingerprint: str,
    file_record: FileReadResult | None = None,
) -> PreparedFileBuild | None:
    if file_record is None:
        file_record = read_changed_file(context.notes_dir, path, context.logger)
    if not file_record:
        context.logger.warning(f"      ⚠️ 跳过空文件或读取失败文件：{path}")
        return None

    file_chunk_texts, file_chunk_meta = _build_chunk_payloads(file_record.doc, path)
    return PreparedFileBuild(
        path=path,
        fingerprint=fingerprint,
        file_record=file_record,
        shadow_tags="",
        scene_tags="",
        scene_tags_version=_SCENE_TAG_VERSION,
        chunk_texts=file_chunk_texts,
        chunk_meta=file_chunk_meta,
    )
