from __future__ import annotations

import datetime
from typing import Any

import numpy as np

from retrieval.chunking import chunk_text
from retrieval.repo_index_types import PreparedFileBuild, RepoState, ScannedRepo


def _make_enhanced_doc(doc: str, shadow_tags: str, scene_tags: str) -> str:
    prefix_lines: list[str] = []
    if scene_tags:
        prefix_lines.append(f"【用途场景：{scene_tags}】")
    if shadow_tags:
        prefix_lines.append(f"【核心隐含特征：{shadow_tags}】")
    if prefix_lines:
        return "\n".join(prefix_lines) + "\n" + doc
    return doc


def _encode_doc_embedding(context, doc: str, shadow_tags: str, scene_tags: str, path: str):
    context.logger.info(f"   🧪 正在编码全文向量：{path}")
    enhanced_doc = _make_enhanced_doc(doc, shadow_tags, scene_tags)
    return _model_encode(context.model_emb, [enhanced_doc], batch_size=1)[0]


def _build_chunk_payloads(doc: str, path: str) -> tuple[list[str], list[dict]]:
    chunks = chunk_text(doc, chunk_size=1000, overlap=200)
    file_chunk_texts = [chunk["text"] for chunk in chunks]
    file_chunk_meta = [
        {"path": path, "chunk_id": idx, "start": chunk["start"], "end": chunk["end"]}
        for idx, chunk in enumerate(chunks)
    ]
    return file_chunk_texts, file_chunk_meta


def _make_enhanced_chunk_texts(chunk_text_list: list[str], path: str, meta_list: list[dict]) -> list[str]:
    return [
        f"【所属文件：{path}~chunk:{meta['chunk_id']}】\n{chunk_text_item}"
        for chunk_text_item, meta in zip(chunk_text_list, meta_list)
    ]


def _encode_chunk_embeddings(context, file_chunk_texts: list[str], file_chunk_meta: list[dict], path: str):
    enhanced_chunk_texts = _make_enhanced_chunk_texts(file_chunk_texts, path, file_chunk_meta)
    if not enhanced_chunk_texts:
        return np.empty((0, _get_embedding_dim(context.model_emb)), dtype=np.float32)

    context.logger.info(f"   🧪 正在编码 chunk 向量：{path}（{len(enhanced_chunk_texts)} 段）")
    return _model_encode(context.model_emb, enhanced_chunk_texts, batch_size=len(enhanced_chunk_texts))


def _get_embedding_dim(model_emb) -> int:
    dim_getter = getattr(model_emb, "get_sentence_embedding_dimension", None)
    if not callable(dim_getter):
        return 0
    try:
        return int(dim_getter() or 0)
    except Exception:
        return 0


def _model_encode(model_emb, texts: list[str], batch_size: int):
    if not texts:
        return np.empty((0, _get_embedding_dim(model_emb)), dtype=np.float32)

    try:
        encoded = model_emb.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    except TypeError:
        encoded = model_emb.encode(texts)
    return np.asarray(encoded)


def build_cache_entries_from_prepared(
    context,
    prepared_files: list[PreparedFileBuild],
    embed_batch_size: int,
) -> dict[str, tuple[dict, dict]]:
    if not prepared_files:
        return {}

    enhanced_docs = [
        _make_enhanced_doc(prepared.file_record.doc, prepared.shadow_tags, prepared.scene_tags)
        for prepared in prepared_files
    ]
    context.logger.info(f"   🧪 全文编码进度 [0/{len(prepared_files)}] 0%")
    context.logger.info(
        f"   🧪 批量编码全文向量：{len(prepared_files)} 个文件（batch_size={embed_batch_size}）"
    )
    doc_embeddings = _model_encode(context.model_emb, enhanced_docs, batch_size=embed_batch_size)
    context.logger.info(f"   🧪 全文编码进度 [{len(prepared_files)}/{len(prepared_files)}] 100%")

    enhanced_chunk_texts: list[str] = []
    chunk_ranges: dict[str, tuple[int, int]] = {}
    for prepared in prepared_files:
        start = len(enhanced_chunk_texts)
        enhanced_chunk_texts.extend(_make_enhanced_chunk_texts(prepared.chunk_texts, prepared.path, prepared.chunk_meta))
        chunk_ranges[prepared.path] = (start, len(enhanced_chunk_texts))

    if enhanced_chunk_texts:
        context.logger.info(f"   🧪 chunk编码进度 [0/{len(prepared_files)}] 0%")
        context.logger.info(
            f"   🧪 批量编码 chunk 向量：{len(prepared_files)} 个文件（{len(enhanced_chunk_texts)} 段, batch_size={embed_batch_size}）"
        )
        chunk_embeddings = _model_encode(context.model_emb, enhanced_chunk_texts, batch_size=embed_batch_size)
        context.logger.info(f"   🧪 chunk编码进度 [{len(prepared_files)}/{len(prepared_files)}] 100%")
    else:
        chunk_embeddings = np.empty((0, _get_embedding_dim(context.model_emb)), dtype=np.float32)

    cache_entries: dict[str, tuple[dict, dict]] = {}
    for idx, prepared in enumerate(prepared_files):
        chunk_start, chunk_end = chunk_ranges[prepared.path]
        file_record = prepared.file_record
        doc_cache_entry = {
            "fingerprint": prepared.fingerprint,
            "doc": file_record.doc,
            "file_time": file_record.file_time.timestamp(),
            "file_size": int(file_record.file_size),
            "file_info": file_record.file_info,
            "shadow_tags": prepared.shadow_tags,
            "scene_tags": prepared.scene_tags,
            "scene_tags_version": int(prepared.scene_tags_version),
            "embedding": np.asarray(doc_embeddings[idx]),
        }
        chunk_cache_entry = {
            "fingerprint": prepared.fingerprint,
            "chunk_texts": prepared.chunk_texts,
            "chunk_meta": prepared.chunk_meta,
            "chunk_file_time": file_record.file_time.timestamp(),
            "chunk_embeddings": np.asarray(chunk_embeddings[chunk_start:chunk_end]),
        }
        cache_entries[prepared.path] = (doc_cache_entry, chunk_cache_entry)

    return cache_entries


def assemble_repo_state(
    scanned: ScannedRepo,
    current_paths: list[str],
    new_doc_cache: dict[str, dict],
    new_chunk_cache: dict[str, dict],
) -> RepoState:
    docs: list[str] = []
    doc_records: list[dict] = []
    file_times: list[datetime.datetime] = []
    file_info_list: list[str] = []
    embeddings_list: list[Any] = []
    chunk_texts: list[str] = []
    chunk_paths: list[str] = []
    chunk_meta: list[dict] = []
    chunk_file_times: list[datetime.datetime] = []
    chunk_embeddings_parts: list[Any] = []
    final_paths: list[str] = []

    for path in current_paths:
        doc_entry = new_doc_cache.get(path)
        chunk_entry = new_chunk_cache.get(path)
        if not doc_entry or not chunk_entry:
            continue

        final_paths.append(path)
        docs.append(doc_entry["doc"])

        doc_time = datetime.datetime.fromtimestamp(float(doc_entry["file_time"]))
        doc_records.append(
            {
                "path": path,
                "shadow_tags": doc_entry.get("shadow_tags", "") or "",
                "scene_tags": doc_entry.get("scene_tags", "") or "",
                "file_time": doc_time,
                "file_size": int(doc_entry.get("file_size", 0) or 0),
                "file_info": doc_entry["file_info"],
            }
        )

        file_times.append(doc_time)
        file_info_list.append(doc_entry["file_info"])
        embeddings_list.append(np.asarray(doc_entry["embedding"]))

        file_chunk_texts = list(chunk_entry["chunk_texts"])
        file_chunk_meta = list(chunk_entry["chunk_meta"])
        file_chunk_time = datetime.datetime.fromtimestamp(float(chunk_entry["chunk_file_time"]))
        file_chunk_embeddings = np.asarray(chunk_entry["chunk_embeddings"])

        chunk_texts.extend(file_chunk_texts)
        chunk_paths.extend([path] * len(file_chunk_texts))
        chunk_meta.extend(file_chunk_meta)
        chunk_file_times.extend([file_chunk_time] * len(file_chunk_texts))

        if file_chunk_texts:
            chunk_embeddings_parts.append(file_chunk_embeddings)

    if embeddings_list:
        embeddings = np.vstack(embeddings_list)
    else:
        embeddings = np.empty((0, 0), dtype=np.float32)

    if chunk_embeddings_parts:
        chunk_embeddings = np.vstack(chunk_embeddings_parts)
    else:
        inferred_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else 0
        chunk_embeddings = np.empty((0, inferred_dim), dtype=np.float32)

    return RepoState(
        docs=docs,
        doc_records=doc_records,
        paths=final_paths,
        file_times=file_times,
        file_info_list=file_info_list,
        chunk_texts=chunk_texts,
        chunk_paths=chunk_paths,
        chunk_meta=chunk_meta,
        chunk_file_times=chunk_file_times,
        all_files=scanned.all_files,
        embeddings=embeddings,
        chunk_embeddings=chunk_embeddings,
        earliest_note=scanned.earliest_note,
        latest_note=scanned.latest_note,
    )
