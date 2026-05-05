from __future__ import annotations

import json
import math
import re
from collections import Counter

from retrieval.repo_index_types import PreparedFileBuild

_MAX_SHADOW_TAGS = 8
_BAD_SHADOW_PREFIXES = ["无法确定", "无法识别", "请提供", "以下是", "根据文本", "关键词如下", "核心关键词如下"]
_SHADOW_TAG_STOPWORDS = {"关键词", "核心关键词", "文本", "内容", "生活类", "技术类", "游戏类"}

_MAX_SCENE_TAGS = 4
_SCENE_TAG_VERSION = 2
_BAD_SCENE_PREFIXES = ["无法确定", "无法识别", "请提供", "以下是", "根据文本", "场景标签如下", "用途标签如下"]
_SCENE_TAG_STOPWORDS = {"场景", "用途", "主题", "内容", "文档", "资料", "知识库", "标签", "关键词", "材料类型"}

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
