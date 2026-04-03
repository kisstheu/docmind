from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any, List

import numpy as np
import requests

from loaders.file_loader import read_file
from retrieval.chunking import chunk_text
from retrieval.query_utils import SUPPORTED_EXT
from retrieval.repo_index_types import FileReadResult, IndexBuildContext, RepoState, ScanEntry, ScannedRepo

_EXCLUDED_PARTS = {
    ".venv",
    ".idea",
    ".git",
    ".SynologyWorkingDirectory",
    "__pycache__",
    ".docmind_trash",
}
_MAX_FILE_BYTES = 500 * 1024
_MAX_IMAGE_FILE_BYTES = 5 * 1024 * 1024
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_MAX_SHADOW_TAGS = 8
_BAD_SHADOW_PREFIXES = ["无法确定", "无法识别", "请提供", "以下是", "根据文本", "关键词如下", "核心关键词如下"]
_SHADOW_TAG_STOPWORDS = {"关键词", "核心关键词", "文本", "内容", "生活类", "技术类", "游戏类"}
_MAX_SCENE_TAGS = 4
_SCENE_TAG_VERSION = 2
_BAD_SCENE_PREFIXES = ["无法确定", "无法识别", "请提供", "以下是", "根据文本", "场景标签如下", "用途标签如下"]
_SCENE_TAG_STOPWORDS = {"场景", "用途", "主题", "内容", "文档", "资料", "知识库", "标签", "关键词", "材料类型"}
_OLLAMA_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}


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
    if file.name.endswith(".ocr.txt") or file.name.startswith("~$") or file.name.endswith(".converted.txt"):
        return False
    stat = file.stat()
    if stat.st_size == 0:
        return False
    size_limit = _MAX_IMAGE_FILE_BYTES if file.suffix.lower() in _IMAGE_EXTENSIONS else _MAX_FILE_BYTES
    if stat.st_size > size_limit:
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


def clean_scene_tags(raw: str) -> str:
    if not raw:
        return ""

    text = raw.strip().replace("\r", "\n")
    for prefix in _BAD_SCENE_PREFIXES:
        if text.startswith(prefix):
            return ""

    parts = re.split(r"[\s,，、;；|]+", text)
    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        token = re.sub(r"^\s*\d+[.、]\s*", "", part).strip(" []()（）-")
        if not token:
            continue
        if token in _SCENE_TAG_STOPWORDS:
            continue
        if token.startswith("材料类型提示"):
            continue
        if token.endswith("标签如下") or token.endswith("关键词如下"):
            continue
        if token.startswith("场景") and len(token) <= 4:
            continue
        if token.startswith("用途") and len(token) <= 4:
            continue
        if len(token) < 2 or len(token) > 16:
            continue

        token = _canonicalize_scene_tag(token)
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(token)

    return " ".join(cleaned[:_MAX_SCENE_TAGS])


def _canonicalize_scene_tag(tag: str) -> str:
    t = (tag or "").strip()
    if not t:
        return ""

    lowered = t.lower()
    if any(x in t for x in ("岗位职责", "任职要求", "岗位要求", "职位描述", "招聘")) or lowered in {"jd", "job description"}:
        return "招聘岗位信息"
    if ("会议" in t and any(x in t for x in ("纪要", "议题", "结论"))) or t == "会议纪要":
        return "会议纪要"
    if any(x in t for x in ("复盘", "回顾", "根因", "改进项")):
        return "项目复盘"
    if any(x in t for x in ("学习笔记", "教程", "课程", "知识点", "读书笔记")):
        return "学习笔记"

    return t


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



def build_changed_file_cache_entry(
    context: IndexBuildContext,
    path: str,
    fingerprint: str,
    file_record: FileReadResult | None = None,
) -> tuple[dict, dict] | None:
    if file_record is None:
        file_record = read_changed_file(context.notes_dir, path, context.logger)
    if not file_record:
        context.logger.warning(f"      ⚠️ 跳过空文件或读取失败文件：{path}")
        return None

    shadow_tags, scene_tags = _extract_tags_for_indexing(context, file_record.doc, path)
    doc_embedding = _encode_doc_embedding(context, file_record.doc, shadow_tags, scene_tags, path)

    file_chunk_texts, file_chunk_meta = _build_chunk_payloads(file_record.doc, path)
    file_chunk_embeddings = _encode_chunk_embeddings(context, file_chunk_texts, file_chunk_meta, path)

    doc_cache_entry = {
        "fingerprint": fingerprint,
        "doc": file_record.doc,
        "file_time": file_record.file_time.timestamp(),
        "file_size": int(file_record.file_size),
        "file_info": file_record.file_info,
        "shadow_tags": shadow_tags,
        "scene_tags": scene_tags,
        "scene_tags_version": _SCENE_TAG_VERSION,
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
            context.logger.info(f"      ✨ 提取到影子标签：[{shadow_tags}]")
        if scene_tags:
            context.logger.info(f"      🧭 提取到场景标签：[{scene_tags}]")
        return shadow_tags, scene_tags
    except Exception as e:
        context.logger.warning(f"      ⚠️ {path} 透视失败，使用空标签 ({e})")
        return "", ""


def _build_combined_tag_prompt(doc: str) -> str:
    return (
        "请从下面文本中提取两类标签，并严格按两行输出：\n"
        "影子标签: 5-8个关键词，空格分隔\n"
        "场景标签: 1-3个标签，空格分隔，第一项优先写材料类型\n"
        "不要输出其他解释。\n"
        + f"文本：\n{doc[:1600]}"
    )


def _parse_combined_tag_response(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()
    if not text:
        return "", ""

    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            return str(data.get("shadow_tags", "") or ""), str(data.get("scene_tags", "") or "")
        except Exception:
            pass

    shadow_raw = ""
    scene_raw = ""
    shadow_match = re.search(r"(?:^|\n)\s*(?:影子标签|细标签|关键词|shadow_tags?)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if shadow_match:
        shadow_raw = shadow_match.group(1).strip()

    scene_match = re.search(r"(?:^|\n)\s*(?:场景标签|用途标签|scene_tags?)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if scene_match:
        scene_raw = scene_match.group(1).strip()

    if shadow_raw or scene_raw:
        return shadow_raw, scene_raw

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) >= 2:
        return lines[0], lines[1]
    if len(lines) == 1:
        return lines[0], ""
    return "", ""


def _request_ollama_response(context: IndexBuildContext, prompt: str, path: str, purpose: str) -> str:
    max_attempts = max(1, int(getattr(context, "ollama_max_retries", 0) or 0) + 1)
    timeout_sec = float(getattr(context, "ollama_timeout_sec", 8.0) or 8.0)
    payload = {"model": context.ollama_model, "prompt": prompt, "stream": False}
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


def _make_enhanced_doc(doc: str, shadow_tags: str, scene_tags: str) -> str:
    prefix_lines: list[str] = []
    if scene_tags:
        prefix_lines.append(f"【用途场景：{scene_tags}】")
    if shadow_tags:
        prefix_lines.append(f"【核心隐藏特征：{shadow_tags}】")
    if prefix_lines:
        return "\n".join(prefix_lines) + "\n" + doc
    return doc


def _encode_doc_embedding(context: IndexBuildContext, doc: str, shadow_tags: str, scene_tags: str, path: str):
    context.logger.info(f"   🧠 正在编码全文向量：{path}")
    enhanced_doc = _make_enhanced_doc(doc, shadow_tags, scene_tags)
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
        embedding_dim = 0
        dim_getter = getattr(context.model_emb, "get_sentence_embedding_dimension", None)
        if callable(dim_getter):
            try:
                embedding_dim = int(dim_getter() or 0)
            except Exception:
                embedding_dim = 0
        return np.empty((0, embedding_dim), dtype=np.float32)
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

        if len(file_chunk_texts) > 0:
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
