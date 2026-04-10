from __future__ import annotations

import math
import datetime
import json
import re
from collections import Counter
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
from retrieval.repo_index_types import FileReadResult, IndexBuildContext, PreparedFileBuild, ScanEntry, ScannedRepo

_MAX_SHADOW_TAGS = 8
_BAD_SHADOW_PREFIXES = ["无法确定", "无法识别", "请提供", "以下是", "根据文本", "关键词如下", "核心关键词如下"]
_SHADOW_TAG_STOPWORDS = {"关键词", "核心关键词", "文本", "内容", "生活类", "技术类", "游戏类"}

_MAX_SCENE_TAGS = 4
_SCENE_TAG_VERSION = 2
_BAD_SCENE_PREFIXES = ["无法确定", "无法识别", "请提供", "以下是", "根据文本", "场景标签如下", "用途标签如下"]
_SCENE_TAG_STOPWORDS = {"场景", "用途", "主题", "内容", "文档", "资料", "知识库", "标签", "关键词", "材料类型"}

_OLLAMA_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_DEFAULT_TAG_EXCERPT_CHARS = 700


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
    text = text.replace("，", " ").replace("。", " ").replace(",", " ").replace(";", " ").replace("；", " ")
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


def _canonicalize_scene_tag(tag: str) -> str:
    t = (tag or "").strip()
    if not t:
        return ""

    lowered = t.lower()
    if any(x in t for x in ("岗位职责", "任职要求", "岗位要求", "职位描述", "招聘", "招聘准备")) or lowered in {"jd", "job description"}:
        return "招聘岗位信息"
    if ("会议" in t and any(x in t for x in ("纪要", "议题", "结论"))) or t == "会议纪要":
        return "会议纪要"
    if any(x in t for x in ("复盘", "回顾", "根因", "改进项")):
        return "项目复盘"
    if any(x in t for x in ("学习笔记", "教程", "课程", "知识点", "读书笔记")):
        return "学习笔记"
    return t


def clean_scene_tags(raw: str) -> str:
    if not raw:
        return ""

    text = raw.strip().replace("\r", "\n")
    for prefix in _BAD_SCENE_PREFIXES:
        if text.startswith(prefix):
            return ""

    parts = re.split(r"[\s,，。;；|]+", text)
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


def _build_combined_tag_prompt(doc: str) -> str:
    return (
        "请从下面文本中提取两类标签，并严格按两行输出：\n"
        "影子标签: 5-8个关键词，空格分隔\n"
        "场景标签: 1-3个标签，空格分隔，第一项优先写材料类型\n"
        "不要输出其他解释。\n"
        + f"文本：\n{_build_tag_excerpt(doc)}"
    )


def _build_batch_tag_prompt(batch_items: list[PreparedFileBuild]) -> str:
    parts = [
        "请为每个文件片段提取两类标签，并仅输出 JSON 数组。",
        "每个元素格式为: {\"id\":\"F1\",\"shadow_tags\":\"关键词1 关键词2\",\"scene_tags\":\"标签1 标签2\"}",
        "shadow_tags: 5-8个关键词，空格分隔。",
        "scene_tags: 1-3个标签，空格分隔，第一项优先写材料类型。",
        "不要输出解释，不要输出 markdown 代码块。",
    ]
    for idx, item in enumerate(batch_items, start=1):
        parts.append(f"[F{idx}] path={item.path}")
        parts.append(_build_tag_excerpt(item.file_record.doc))
    return "\n\n".join(parts)


def _build_tag_excerpt(doc: str) -> str:
    text = (doc or "").strip()
    limit = _DEFAULT_TAG_EXCERPT_CHARS
    if len(text) <= limit:
        return text
    if limit <= 240:
        return text[:limit]

    head = int(limit * 0.65)
    tail = max(0, limit - head - 5)
    if tail == 0:
        return text[:limit]
    return f"{text[:head]}\n...\n{text[-tail:]}"


def _candidate_key(text: str) -> str:
    return (text or "").strip().casefold()


def _normalize_candidate(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip("[](){}<>:：,，。；;|/-_~!@#$%^&*+=?\"'`")
    if not cleaned:
        return ""
    if cleaned.isdigit():
        return ""
    if len(cleaned) < 2 or len(cleaned) > 32:
        return ""
    return cleaned


def _extract_token_candidates(doc: str) -> list[str]:
    text = (doc or "").strip()
    if not text:
        return []

    candidates: list[str] = []
    for match in re.finditer(r"[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9+#._/-]{1,31}", text):
        candidate = _normalize_candidate(match.group(0))
        if candidate:
            candidates.append(candidate)
    return candidates


def _extract_line_candidates(doc: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in (doc or "").splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        candidate = _normalize_candidate(line)
        if candidate and len(candidate) <= 18:
            candidates.append(candidate)
    return candidates


def _build_statistical_tag_stats(existing_docs: list[str], prepared_files: list[PreparedFileBuild]) -> dict:
    corpus_docs = [doc for doc in existing_docs if isinstance(doc, str) and doc.strip()]
    corpus_docs.extend(prepared.file_record.doc for prepared in prepared_files if prepared.file_record.doc.strip())
    total_docs = max(1, len(corpus_docs))

    token_df: Counter[str] = Counter()
    line_df: Counter[str] = Counter()
    for doc in corpus_docs:
        token_df.update({_candidate_key(token) for token in _extract_token_candidates(doc)})
        line_df.update({_candidate_key(line) for line in _extract_line_candidates(doc)})

    return {
        "total_docs": total_docs,
        "token_df": token_df,
        "line_df": line_df,
    }


def _idf(total_docs: int, doc_freq: int) -> float:
    return math.log((total_docs + 1) / (doc_freq + 1)) + 1.0


def _score_shadow_tags(doc: str, stats: dict) -> str:
    total_docs = int(stats["total_docs"])
    token_df: Counter[str] = stats["token_df"]
    token_counter: Counter[str] = Counter()
    representative: dict[str, str] = {}

    for token in _extract_token_candidates(doc):
        key = _candidate_key(token)
        if not key:
            continue
        token_counter[key] += 1
        representative.setdefault(key, token)

    scored: list[tuple[float, str]] = []
    for key, freq in token_counter.items():
        token = representative[key]
        score = float(freq) * _idf(total_docs, int(token_df.get(key, 0)))
        score += min(len(token), 12) * 0.03
        if any(ch.isalpha() for ch in token) and any("\u4e00" <= ch <= "\u9fff" for ch in token):
            score += 0.08
        scored.append((score, token))

    scored.sort(key=lambda item: (-item[0], -len(item[1]), item[1]))
    return clean_shadow_tags(" ".join(token for _score, token in scored[:_MAX_SHADOW_TAGS]))


def _score_scene_tags(doc: str, stats: dict) -> str:
    total_docs = int(stats["total_docs"])
    line_df: Counter[str] = stats["line_df"]
    candidates = _extract_line_candidates(doc)
    if not candidates:
        candidates = [token for token in _extract_token_candidates(doc) if 2 <= len(token) <= 16]

    seen: set[str] = set()
    scored: list[tuple[float, str]] = []
    for candidate in candidates:
        key = _candidate_key(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        score = _idf(total_docs, int(line_df.get(key, 0)))
        score += min(len(candidate), 12) * 0.02
        scored.append((score, candidate))

    scored.sort(key=lambda item: (-item[0], -len(item[1]), item[1]))
    return clean_scene_tags(" ".join(candidate for _score, candidate in scored[:_MAX_SCENE_TAGS]))


def _extract_statistical_tags_for_indexing(doc: str, stats: dict) -> tuple[str, str]:
    return _score_shadow_tags(doc, stats), _score_scene_tags(doc, stats)


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
    shadow_match = re.search(r"(?:^|\n)\s*(?:影子标签|shadow_tags?)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if shadow_match:
        shadow_raw = shadow_match.group(1).strip()

    scene_match = re.search(r"(?:^|\n)\s*(?:场景标签|scene_tags?)\s*[:：]\s*(.+)", text, flags=re.IGNORECASE)
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


def _parse_batch_tag_response(raw: str) -> dict[str, tuple[str, str]]:
    text = (raw or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    parsed = json.loads(text)
    items = parsed.get("items", []) if isinstance(parsed, dict) else parsed
    results: dict[str, tuple[str, str]] = {}
    if not isinstance(items, list):
        return results

    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id", "") or "").strip()
        if not item_id:
            continue
        results[item_id] = (
            str(item.get("shadow_tags", "") or "").strip(),
            str(item.get("scene_tags", "") or "").strip(),
        )
    return results


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
