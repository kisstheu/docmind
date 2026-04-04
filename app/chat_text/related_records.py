from __future__ import annotations

import re
from pathlib import Path

from app.chat_text.core import is_related_record_listing_request

def extract_related_topic(question: str) -> str:
    q = re.sub(r"\s+", "", (question or "").strip())
    if not q:
        return ""

    patterns = [
        r"(?:最近)?有哪些?和(.+?)(?:有关|相关)(?:的)?(?:记录|文档|文件)",
        r"(.+?)(?:有关|相关)(?:的)?(?:记录|文档|文件)有哪些?",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if not m:
            continue
        topic = (m.group(1) or "").strip("，。！？；：,.!?;:")
        topic = re.sub(r"^(和|与|跟|关于)", "", topic)
        return topic
    return ""


def _build_topic_variants(topic: str) -> list[str]:
    t = (topic or "").strip().lower()
    if not t:
        return []

    variants: list[str] = []

    def _add(x: str):
        s = (x or "").strip().lower()
        if s and s not in variants:
            variants.append(s)

    _add(t)

    for token in re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fa5]{2,}", t):
        _add(token)

    if re.fullmatch(r"[\u4e00-\u9fa5]{4,}", t):
        _add(t[:2])
        _add(t[-2:])
        for i in range(len(t) - 1):
            _add(t[i:i + 2])

    return variants


def maybe_build_related_records_answer(question: str, relevant_indices, repo_state, max_items: int = 8) -> str | None:
    if not is_related_record_listing_request(question):
        return None

    topic = extract_related_topic(question)
    if not topic:
        return None

    variants = _build_topic_variants(topic)
    if not variants:
        return None

    file_hits: dict[str, dict] = {}

    for idx in relevant_indices or []:
        path = repo_state.chunk_paths[idx]
        text = repo_state.chunk_texts[idx] or ""
        path_lower = path.lower()
        text_lower = text.lower()

        score = 0
        for v in variants:
            if v in path_lower:
                score = max(score, 3)
            if v in text_lower:
                score = max(score, 2)

        if score <= 0:
            continue

        evidence = ""
        for line in text.splitlines():
            ln = (line or "").strip()
            if not ln:
                continue
            if any(v in ln.lower() for v in variants):
                evidence = ln
                break

        rec = file_hits.get(path)
        dt = repo_state.chunk_file_times[idx]
        if not rec or score > rec["score"]:
            file_hits[path] = {
                "path": path,
                "score": score,
                "dt": dt,
                "evidence": evidence,
            }

    if not file_hits:
        return None

    items = [x for x in file_hits.values() if x["score"] >= 2]
    if not items:
        items = sorted(file_hits.values(), key=lambda x: x["score"], reverse=True)[:3]

    items = sorted(items, key=lambda x: x["dt"], reverse=True)[:max_items]

    lines = [f"根据现有记录，和“{topic}”有关的记录有 {len(items)} 条："]
    for i, item in enumerate(items, 1):
        path = item["path"]
        date_match = re.search(r"(20\d{2}-\d{2}-\d{2})", Path(path).name)
        date_text = date_match.group(1) if date_match else item["dt"].strftime("%Y-%m-%d")
        lines.append(f"{i}. {path}（{date_text}）")
        if item["evidence"]:
            ev = item["evidence"]
            if len(ev) > 80:
                ev = ev[:80] + "..."
            lines.append(f"   证据：{ev}")

    return "\n".join(lines)
