from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Callable, Sequence

COUNT_TARGET_NOISE_PHRASES = (
    "有多少个文件",
    "有多少个文档",
    "有多少文件",
    "有多少文档",
    "多少个文件",
    "多少个文档",
    "多少文件",
    "多少文档",
    "文件数量",
    "文档数量",
    "文件数",
    "文档数",
    "当前",
    "目前",
    "现在",
    "总共",
    "一共",
    "总计",
    "大概",
    "大约",
    "差不多",
)
GENERIC_COUNT_TARGETS = {
    "这块",
    "这一块",
    "这类",
    "这一类",
    "这些",
    "那些",
    "这个",
    "那个",
    "这里",
    "这边",
    "那边",
    "本地",
    "仓库",
    "知识库",
}

from ai.capability_common import clean_text, normalize_meta_question


def _extract_summary_labels(summary_text: str | None) -> list[str]:
    labels: list[str] = []
    for line in (summary_text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        label = stripped[2:].strip()
        label = re.sub(r"[：:]\s*约\s*\d+\s*个文件$", "", label).strip()
        if label:
            labels.append(label)
    return labels


def _extract_count_target_topic(question: str) -> str:
    target = normalize_meta_question(clean_text(question))
    if not target:
        return ""

    for phrase in sorted(COUNT_TARGET_NOISE_PHRASES, key=len, reverse=True):
        target = target.replace(phrase, " ")

    target = re.sub(r"[？?，,。.!！、；;：:\s]+", " ", target)
    target = target.strip(" 的")
    target = re.sub(r"(呢|啊|呀|吗|么|吧)$", "", target).strip()
    if not target or target in GENERIC_COUNT_TARGETS:
        return ""
    return target


def _match_category_label(target_topic: str, category_labels: Sequence[str]) -> str | None:
    target = str(target_topic or "").strip()
    if not target:
        return None

    simplified_target = target.replace("相关", "").replace("类别", "").replace("板块", "").strip()
    for label in sorted(category_labels, key=len, reverse=True):
        normalized_label = str(label or "").strip()
        if not normalized_label:
            continue
        simplified_label = normalized_label.replace("相关", "").replace("类别", "").replace("板块", "").strip()
        if target == normalized_label or target in normalized_label or normalized_label in target:
            return normalized_label
        if simplified_target and simplified_label and (
            simplified_target == simplified_label
            or simplified_target in simplified_label
            or simplified_label in simplified_target
        ):
            return normalized_label
    return None


def _match_category_label_with_local_llm(
    target_topic: str,
    category_labels: Sequence[str],
    topic_summarizer: Callable[[str], str] | None,
) -> str | None:
    if not topic_summarizer:
        return None

    labels = [str(label or "").strip() for label in category_labels if str(label or "").strip()]
    if not labels:
        return None

    prompt = (
        "下面是知识库里已经确定好的板块名，请从中选出和用户说法最接近的一个。\n"
        "要求：\n"
        "1. 只能输出候选里的原词\n"
        "2. 如果没有明显对应，也只输出最接近的一个候选\n"
        "3. 不要解释，不要输出其他内容\n\n"
        "候选板块：\n"
        + "\n".join(f"- {label}" for label in labels)
        + f"\n\n用户说法：{target_topic}"
    )

    try:
        raw = topic_summarizer(prompt)
    except Exception:
        return None

    text = _strip_code_fence(raw).strip()
    text = text.splitlines()[0].strip().lstrip("-*0123456789.、 ").strip()
    text = text.strip("[](){}<>【】「」『』“”\"'` ")
    if text in labels:
        return text
    return _match_category_label(text, labels)


def _trim_topic_text(text: str, max_items: int = 5) -> str:
    parts = [part.strip() for part in str(text or "").split() if part.strip()]
    return " ".join(parts[:max_items])


def _build_record_category_hint(record: dict) -> str:
    scene = _trim_topic_text(record.get("scene_tags", ""), max_items=4)
    shadow = _trim_topic_text(record.get("shadow_tags", ""), max_items=5)
    chunks: list[str] = []
    if scene:
        chunks.append(f"场景={scene}")
    if shadow:
        chunks.append(f"特征={shadow}")
    return "；".join(chunks)


def _build_record_category_embedding_text(record: dict) -> str:
    path = str(record.get("path", "") or "").strip()
    stem = Path(path).stem.replace("_", " ").replace("-", " ").strip() if path else ""
    hint = _build_record_category_hint(record)
    parts = [part for part in (stem, hint) if part]
    return " ".join(parts).strip()


def _strip_code_fence(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()
