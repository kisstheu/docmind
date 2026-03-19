from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import numpy as np
import requests

from loaders.file_loader import read_file
from retrieval.chunking import chunk_text
from retrieval.query_utils import SUPPORTED_EXT


@dataclass
class RepoState:
    docs: List[str]
    doc_records: List[dict]
    paths: List[str]
    file_times: List[datetime.datetime]
    file_info_list: List[str]
    chunk_texts: List[str]
    chunk_paths: List[str]
    chunk_meta: List[dict]
    chunk_file_times: List[datetime.datetime]
    all_files: List[Path]
    embeddings: Any
    chunk_embeddings: Any
    earliest_note: str
    latest_note: str


def collect_all_files(notes_dir: Path) -> List[Path]:
    raw_files = list(notes_dir.rglob("*"))
    all_files = []
    for f in raw_files:
        if not f.is_file():
            continue
        if any(part in f.parts for part in {".venv", ".idea", ".git", ".SynologyWorkingDirectory", "__pycache__"}):
            continue
        if f.suffix.lower() not in SUPPORTED_EXT:
            continue
        if f.name.endswith(".ocr.txt") or f.name.startswith("~$"):
            continue
        stat = f.stat()

        if stat.st_size == 0:
            continue

        if stat.st_size > 500 * 1024:
            continue
        all_files.append(f)
    all_files.sort(key=lambda x: x.stat().st_mtime)
    return all_files
def clean_shadow_tags(raw: str) -> str:
    """
    清洗 Ollama 返回的影子标签，尽量只保留真正可检索的关键词。
    """
    if not raw:
        return ""

    text = raw.strip().replace("\r", "\n")

    # 去掉一些常见“解释型废话”
    bad_prefixes = [
        "无法确定",
        "无法识别",
        "请提供",
        "以下是",
        "根据文本",
        "关键词如下",
        "核心关键词如下",
    ]
    for prefix in bad_prefixes:
        if text.startswith(prefix):
            return ""

    # 去掉分类前缀
    text = re.sub(r"^\s*生活类关键词[:：]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*技术类关键词[:：]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*游戏类关键词[:：]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*生活类[:：]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*技术类[:：]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*游戏类[:：]?\s*", "", text, flags=re.IGNORECASE)

    # 去掉方括号，统一分隔符
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace("，", " ").replace("、", " ").replace(",", " ")
    text = text.replace("；", " ").replace(";", " ")
    text = text.replace("\n", " ").replace("\t", " ")

    # 拆词
    parts = [p.strip() for p in text.split(" ") if p.strip()]

    cleaned = []
    seen = set()

    for p in parts:
        if len(p) > 60:
            continue
        if p in {"关键词", "核心关键词", "文本", "内容", "生活类", "技术类", "游戏类"}:
            continue
        if p.startswith("无法") or p.startswith("请提供"):
            continue

        key = p.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(p)

    # 最多保留 8 个
    return " ".join(cleaned[:8])
def read_changed_file(notes_dir: Path, relative_path: str, logger):
    """
    只读取发生变化的文件内容。
    """
    file_path = notes_dir / relative_path
    stat = file_path.stat()
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
    size_kb = stat.st_size / 1024

    content, used_sidecar = read_file(file_path, logger=logger)
    if not content:
        return None

    return {
        "path": relative_path,
        "doc": content,
        "file_time": mtime,
        "file_info": f"- {relative_path} (大小: {size_kb:.1f}KB, 更新于: {mtime.strftime('%Y-%m-%d')})",
        "used_sidecar": used_sidecar,
    }
def build_file_fingerprint(relative_path: str, stat_result) -> str:
    return f"{relative_path}|{stat_result.st_size}|{int(stat_result.st_mtime_ns)}"


def scan_repository(notes_dir: Path, logger):
    """
    轻量扫描：
    - 只收集路径 / mtime / size / fingerprint
    - 不读取文件正文
    """
    all_files = collect_all_files(notes_dir)

    entries = []
    file_info_list = []

    for file in all_files:
        stat = file.stat()
        mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
        date_str = mtime.strftime("%Y-%m-%d")
        size_kb = stat.st_size / 1024
        relative_path = file.relative_to(notes_dir).as_posix()
        fingerprint = build_file_fingerprint(relative_path, stat)

        entries.append({
            "path": relative_path,
            "file_time": mtime,
            "fingerprint": fingerprint,
            "size_kb": size_kb,
        })
        file_info_list.append(f"- {relative_path} (大小: {size_kb:.1f}KB, 更新于: {date_str})")

    earliest_note = file_info_list[0] if file_info_list else "无"
    latest_note = file_info_list[-1] if file_info_list else "无"

    return {
        "entries": entries,
        "paths": [e["path"] for e in entries],
        "file_times": [e["file_time"] for e in entries],
        "file_info_list": file_info_list,
        "all_files": all_files,
        "earliest_note": earliest_note,
        "latest_note": latest_note,
        "notes_dir": notes_dir,
    }

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


def _extract_shadow_tags(doc: str, path: str, logger, ollama_api_url: str, ollama_model: str) -> str:
    logger.info(f"   🤖 正在透视文件：{path} ...")
    try:
        summary_prompt = (
            f"请提取以下私人笔记片段的 5-8 个最核心搜索关键词。\n"
            f"【严格分类指令】：\n"
            f"1. [技术/职场类]：【仅当】内容明确涉及代码、软件开发、公司事务时，才允许加入“项目”、“工作”及技术词。\n"
            f"2. [游戏/娱乐类]：【仅当】明确涉及游戏时，才加入“游戏”、“个人爱好”及具体游戏名。\n"
            f"3. [生活类]：如果涉及生活琐事、情感日常，【绝对禁止】加入任何技术、代码或工作词汇！提取其本身的专属词即可。\n"
            f"【输出格式】：极度简练，只返回空格分隔的关键词，不许废话。\n\n"
            f"文本：\n{doc[:1500]}"
        )
        payload = {"model": ollama_model, "prompt": summary_prompt, "stream": False}
        response = requests.post(ollama_api_url, json=payload, timeout=180)
        response.raise_for_status()
        raw_tags = response.json().get("response", "").strip()
        shadow_tags = clean_shadow_tags(raw_tags)

        if shadow_tags:
            logger.info(f"      ✨ 提取到影子标签：[{shadow_tags}]")
        else:
            logger.info(f"      ℹ️ 影子标签已清洗为空，原始输出：[{raw_tags[:80]}]")

        return shadow_tags
    except Exception as e:
        logger.warning(f"      ⚠️ {path} 透视失败，使用空标签 ({e})")
        return ""


def _make_enhanced_doc(doc: str, shadow_tags: str) -> str:
    if shadow_tags:
        return f"【核心隐藏特征：{shadow_tags}】\n{doc}"
    return doc


def _make_enhanced_chunk_texts(chunk_text_list: List[str], path: str, meta_list: List[dict]) -> List[str]:
    enhanced = []
    for chunk_text_item, meta in zip(chunk_text_list, meta_list):
        enhanced.append(f"【所属文件：{path}｜chunk:{meta['chunk_id']}】\n{chunk_text_item}")
    return enhanced


def load_or_build_embeddings(scanned, cache_file: Path, model_emb, logger, ollama_api_url: str, ollama_model: str):
    entries = scanned["entries"]
    current_paths = [e["path"] for e in entries]
    current_file_times = [e["file_time"] for e in entries]
    current_file_fingerprints = [e["fingerprint"] for e in entries]

    old_doc_cache = {}
    old_chunk_cache = {}
    old_manifest = {}
    cache_usable = False

    if cache_file.exists():
        try:
            cache = np.load(cache_file, allow_pickle=True)
            old_manifest = _safe_load_object(cache, "manifest", {})
            old_doc_cache = _safe_load_object(cache, "doc_cache", {})
            old_chunk_cache = _safe_load_object(cache, "chunk_cache", {})

            if isinstance(old_manifest, dict) and isinstance(old_doc_cache, dict) and isinstance(old_chunk_cache, dict):
                cache_usable = True
                logger.info("✨ 检测到增量缓存，将执行差量比对")
            else:
                logger.info("⚠️ 缓存结构不是增量版，将重建一次并升级缓存格式。")
        except Exception as e:
            logger.warning(f"⚠️ 读取缓存失败，将重建缓存：{e}")

    current_manifest = {
        path: fp
        for path, fp in zip(current_paths, current_file_fingerprints)
    }

    added_paths = []
    modified_paths = []
    unchanged_paths = []

    if cache_usable:
        for path in current_paths:
            new_fp = current_manifest[path]
            old_fp = old_manifest.get(path)
            if old_fp is None:
                added_paths.append(path)
            elif old_fp != new_fp:
                modified_paths.append(path)
            else:
                unchanged_paths.append(path)
    else:
        added_paths = list(current_paths)

    deleted_paths = []
    if cache_usable:
        for old_path in old_manifest.keys():
            if old_path not in current_manifest:
                deleted_paths.append(old_path)

    logger.info(
        f"📊 本轮索引差量统计：新增 {len(added_paths)} / 修改 {len(modified_paths)} / 删除 {len(deleted_paths)} / 复用 {len(unchanged_paths)}"
    )

    path_to_time = {e["path"]: e["file_time"] for e in entries}
    path_to_fp = {e["path"]: e["fingerprint"] for e in entries}

    new_doc_cache = {}
    new_chunk_cache = {}

    # 先复用未变化文件
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
            modified_paths.append(path)

    if reused_count:
        logger.info(f"♻️ 已复用 {reused_count} 个未变化文件的索引结果")

    changed_paths = []
    seen = set()
    for p in added_paths + modified_paths:
        if p not in seen:
            changed_paths.append(p)
            seen.add(p)

    if changed_paths:
        logger.info("\n🧠 检测到新增或修改文件，开始增量建库...\n")

    sidecar_count = 0
    sidecar_examples = []

    for path in changed_paths:
        file_record = read_changed_file(scanned["notes_dir"], path, logger)
        if not file_record:
            logger.warning(f"      ⚠️ 跳过空文件或读取失败文件：{path}")
            continue

        doc = file_record["doc"]
        file_time = file_record["file_time"]
        fingerprint = path_to_fp[path]

        if file_record["used_sidecar"]:
            sidecar_count += 1
            if len(sidecar_examples) < 5:
                sidecar_examples.append(path + ".ocr.txt")

        shadow_tags = _extract_shadow_tags(doc, path, logger, ollama_api_url, ollama_model)
        enhanced_doc = _make_enhanced_doc(doc, shadow_tags)

        logger.info(f"   🧠 正在编码全文向量：{path}")
        doc_embedding = model_emb.encode([enhanced_doc])[0]

        chunks = chunk_text(doc, chunk_size=1000, overlap=200)
        file_chunk_texts = [ch["text"] for ch in chunks]
        file_chunk_meta = [
            {
                "path": path,
                "chunk_id": idx,
                "start": ch["start"],
                "end": ch["end"],
            }
            for idx, ch in enumerate(chunks)
        ]

        enhanced_chunk_texts = _make_enhanced_chunk_texts(file_chunk_texts, path, file_chunk_meta)

        if enhanced_chunk_texts:
            logger.info(f"   🧠 正在编码 chunk 向量：{path}（{len(enhanced_chunk_texts)} 段）")
            file_chunk_embeddings = model_emb.encode(enhanced_chunk_texts)
        else:
            file_chunk_embeddings = np.empty((0,))

        new_doc_cache[path] = {
            "fingerprint": fingerprint,
            "doc": doc,
            "file_time": file_time.timestamp(),
            "file_info": file_record["file_info"],
            "shadow_tags": shadow_tags,
            "embedding": np.asarray(doc_embedding),
        }

        new_chunk_cache[path] = {
            "fingerprint": fingerprint,
            "chunk_texts": file_chunk_texts,
            "chunk_meta": file_chunk_meta,
            "chunk_file_time": file_time.timestamp(),
            "chunk_embeddings": np.asarray(file_chunk_embeddings),
        }
    if sidecar_count > 0:
        logger.info(f"      📄 本轮变动文件中共命中 {sidecar_count} 个伴生文件")
        if sidecar_examples:
            logger.info(f"         例如: {', '.join(sidecar_examples)}")
    docs = []
    doc_records = []
    file_times = []
    file_info_list = []
    embeddings_list = []
    chunk_texts = []
    chunk_paths = []
    chunk_meta = []
    chunk_file_times = []
    chunk_embeddings_parts = []

    final_paths = []
    for path in current_paths:
        doc_entry = new_doc_cache.get(path)
        chunk_entry = new_chunk_cache.get(path)

        if not doc_entry or not chunk_entry:
            logger.warning(f"      ⚠️ 文件缺少有效索引，已从当前库中跳过：{path}")
            continue

        final_paths.append(path)
        docs.append(doc_entry["doc"])

        doc_records.append({
            "path": path,
            "shadow_tags": doc_entry.get("shadow_tags", "") or "",
            "file_time": datetime.datetime.fromtimestamp(float(doc_entry["file_time"])),
            "file_info": doc_entry["file_info"],
        })

        file_times.append(datetime.datetime.fromtimestamp(float(doc_entry["file_time"])))
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
        embeddings = np.empty((0,))

    if chunk_embeddings_parts:
        chunk_embeddings = np.vstack(chunk_embeddings_parts)
    else:
        chunk_embeddings = np.empty((0,))

    # 保存新缓存（已自动移除 deleted paths）
    np.savez(
        cache_file,
        manifest=np.array(current_manifest, dtype=object),
        doc_cache=np.array(new_doc_cache, dtype=object),
        chunk_cache=np.array(new_chunk_cache, dtype=object),
    )

    if changed_paths or deleted_paths:
        logger.info("✅ 增量索引更新完成")
    else:
        logger.info("✅ 所有索引均直接复用，无需重建")

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
        all_files=scanned["all_files"],
        embeddings=embeddings,
        chunk_embeddings=chunk_embeddings,
        earliest_note=scanned["earliest_note"],
        latest_note=scanned["latest_note"],
    )