from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import requests

from loaders.file_loader import read_file
from retrieval.chunking import chunk_text
from retrieval.query_utils import SUPPORTED_EXT
from retrieval.repo_index_types import FileReadResult, IndexBuildContext, RepoState, ScanEntry, ScannedRepo

_EXCLUDED_PARTS = {".venv", ".idea", ".git", ".SynologyWorkingDirectory", "__pycache__"}
_MAX_FILE_BYTES = 500 * 1024
_MAX_SHADOW_TAGS = 8
_BAD_SHADOW_PREFIXES = ["无法确定", "无法识别", "请提供", "以下是", "根据文本", "关键词如下", "核心关键词如下"]
_SHADOW_TAG_STOPWORDS = {"关键词", "核心关键词", "文本", "内容", "生活类", "技术类", "游戏类"}


def collect_all_files(notes_dir: Path) -> List[Path]:
    all_files: List[Path] = []
    for file in notes_dir.rglob("*"):
        if not _is_supported_file(file):
            continue
        all_files.append(file)
    all_files.sort(key=lambda x: x.stat().st_mtime)
    return all_files



def _is_supported_file(file: Path) -> bool:
    if not file.is_file():
        return False
    if any(part in file.parts for part in _EXCLUDED_PARTS):
        return False
    if file.suffix.lower() not in SUPPORTED_EXT:
        return False
    if file.name.endswith(".ocr.txt") or file.name.startswith("~$"):
        return False
    stat = file.stat()
    if stat.st_size == 0:
        return False
    if stat.st_size > _MAX_FILE_BYTES:
        return False
    return True



def clean_shadow_tags(raw: str) -> str:
    if not raw:
        return ""

    text = raw.strip().replace("\r", "\n")
    for prefix in _BAD_SHADOW_PREFIXES:
        if text.startswith(prefix):
            return ""

    for pattern in [
        r"^\s*生活类关键词[:：]?\s*",
        r"^\s*技术类关键词[:：]?\s*",
        r"^\s*游戏类关键词[:：]?\s*",
        r"^\s*生活类[:：]?\s*",
        r"^\s*技术类[:：]?\s*",
        r"^\s*游戏类[:：]?\s*",
    ]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("，", " ").replace("、", " ").replace(",", " ")
    text = text.replace("；", " ").replace(";", " ")
    text = text.replace("\n", " ").replace("\t", " ")

    cleaned: list[str] = []
    seen: set[str] = set()
    for part in [p.strip() for p in text.split(" ") if p.strip()]:
        if len(part) > 20 and part.count("_") > 2:
            continue
        if len(part) > 60:
            continue
        if part in _SHADOW_TAG_STOPWORDS:
            continue
        if part.startswith("无法") or part.startswith("请提供"):
            continue

        key = part.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(part)

    return " ".join(cleaned[:_MAX_SHADOW_TAGS])



def build_file_fingerprint(relative_path: str, stat_result) -> str:
    return f"{relative_path}|{stat_result.st_size}|{int(stat_result.st_mtime_ns)}"



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



def _format_file_info(relative_path: str, size_kb: float, mtime: datetime.datetime) -> str:
    return f"- {relative_path} (大小: {size_kb:.1f}KB, 更新于: {mtime.strftime('%Y-%m-%d')})"



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



def build_changed_file_cache_entry(context: IndexBuildContext, path: str, fingerprint: str) -> tuple[dict, dict] | None:
    file_record = read_changed_file(context.notes_dir, path, context.logger)
    if not file_record:
        context.logger.warning(f"      ⚠️ 跳过空文件或读取失败文件：{path}")
        return None

    shadow_tags = _extract_shadow_tags(context, file_record.doc, path)
    doc_embedding = _encode_doc_embedding(context, file_record.doc, shadow_tags, path)

    file_chunk_texts, file_chunk_meta = _build_chunk_payloads(file_record.doc, path)
    file_chunk_embeddings = _encode_chunk_embeddings(context, file_chunk_texts, file_chunk_meta, path)

    doc_cache_entry = {
        "fingerprint": fingerprint,
        "doc": file_record.doc,
        "file_time": file_record.file_time.timestamp(),
        "file_size": int(file_record.file_size),
        "file_info": file_record.file_info,
        "shadow_tags": shadow_tags,
        "embedding": np.asarray(doc_embedding),
    }
    chunk_cache_entry = {
        "fingerprint": fingerprint,
        "chunk_texts": file_chunk_texts,
        "chunk_meta": file_chunk_meta,
        "chunk_file_time": file_record.file_time.timestamp(),
        "chunk_embeddings": np.asarray(file_chunk_embeddings),
    }
    return doc_cache_entry, chunk_cache_entry



def _extract_shadow_tags(context: IndexBuildContext, doc: str, path: str) -> str:
    context.logger.info(f"   🤖 正在透视文件：{path} ...")
    try:
        payload = {
            "model": context.ollama_model,
            "prompt": _build_shadow_tag_prompt(doc),
            "stream": False,
        }
        response = requests.post(context.ollama_api_url, json=payload, timeout=180)
        response.raise_for_status()
        raw_tags = response.json().get("response", "").strip()
        shadow_tags = clean_shadow_tags(raw_tags)
        if shadow_tags:
            context.logger.info(f"      ✨ 提取到影子标签：[{shadow_tags}]")
        else:
            context.logger.info(f"      ℹ️ 影子标签已清洗为空，原始输出：[{raw_tags[:80]}]")
        return shadow_tags
    except Exception as e:
        context.logger.warning(f"      ⚠️ {path} 透视失败，使用空标签 ({e})")
        return ""



def _build_shadow_tag_prompt(doc: str) -> str:
    return (
        "请提取以下私人笔记片段的 5-8 个最核心搜索关键词。\n"
        "【严格分类指令】：\n"
        "1. [技术/职场类]：【仅当】内容明确涉及代码、软件开发、公司事务时，才允许加入“项目”、“工作”及技术词。\n"
        "2. [游戏/娱乐类]：【仅当】明确涉及游戏时，才加入“游戏”、“个人爱好”及具体游戏名。\n"
        "3. [生活类]：如果涉及生活琐事、情感日常，【绝对禁止】加入任何技术、代码或工作词汇！提取其本身的专属词即可。\n"
        "【输出格式】：极度简练，只返回空格分隔的关键词，不许废话。\n\n"
        f"文本：\n{doc[:1500]}"
    )



def _make_enhanced_doc(doc: str, shadow_tags: str) -> str:
    if shadow_tags:
        return f"【核心隐藏特征：{shadow_tags}】\n{doc}"
    return doc



def _encode_doc_embedding(context: IndexBuildContext, doc: str, shadow_tags: str, path: str):
    context.logger.info(f"   🧠 正在编码全文向量：{path}")
    enhanced_doc = _make_enhanced_doc(doc, shadow_tags)
    return context.model_emb.encode([enhanced_doc])[0]



def _build_chunk_payloads(doc: str, path: str) -> tuple[list[str], list[dict]]:
    chunks = chunk_text(doc, chunk_size=1000, overlap=200)
    file_chunk_texts = [chunk["text"] for chunk in chunks]
    file_chunk_meta = [
        {"path": path, "chunk_id": idx, "start": chunk["start"], "end": chunk["end"]}
        for idx, chunk in enumerate(chunks)
    ]
    return file_chunk_texts, file_chunk_meta



def _make_enhanced_chunk_texts(chunk_text_list: List[str], path: str, meta_list: List[dict]) -> List[str]:
    return [
        f"【所属文件：{path}｜chunk:{meta['chunk_id']}】\n{chunk_text_item}"
        for chunk_text_item, meta in zip(chunk_text_list, meta_list)
    ]



def _encode_chunk_embeddings(context: IndexBuildContext, file_chunk_texts: list[str], file_chunk_meta: list[dict], path: str):
    enhanced_chunk_texts = _make_enhanced_chunk_texts(file_chunk_texts, path, file_chunk_meta)
    if not enhanced_chunk_texts:
        return np.empty((0,))
    context.logger.info(f"   🧠 正在编码 chunk 向量：{path}（{len(enhanced_chunk_texts)} 段）")
    return context.model_emb.encode(enhanced_chunk_texts)



def assemble_repo_state(scanned: ScannedRepo, current_paths: list[str], new_doc_cache: dict[str, dict], new_chunk_cache: dict[str, dict]) -> RepoState:
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

        if len(file_chunk_texts) > 0:
            chunk_embeddings_parts.append(file_chunk_embeddings)

    embeddings = np.vstack(embeddings_list) if embeddings_list else np.empty((0,))
    chunk_embeddings = np.vstack(chunk_embeddings_parts) if chunk_embeddings_parts else np.empty((0,))

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
