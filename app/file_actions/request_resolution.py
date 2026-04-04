from __future__ import annotations

import re

from app.file_actions.common import find_repo_path_by_reference

_NUMBERED_RESULT_LINE_PATTERN = re.compile(r"^\s*(\d{1,3})[.。\)]\s*(.+?)\s*$")
_RESULT_SOURCE_LINE_PATTERN = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:来源|出处|原文件|源文件)\s*[:：]\s*(.+?)\s*$",
    flags=re.IGNORECASE,
)
_FILE_PATH_PATTERN = re.compile(
    r"([A-Za-z0-9_\-\u4e00-\u9fa5\\/:.\s]+?\.(?:txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx|png|jpg|jpeg|bmp|webp))\b",
    flags=re.IGNORECASE,
)
_CN_NUM_MAP = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _parse_cn_number(token: str) -> int | None:
    t = (token or "").strip()
    if not t:
        return None
    if t.isdigit():
        return int(t)
    if t in _CN_NUM_MAP:
        return _CN_NUM_MAP[t]
    if len(t) == 2 and t[0] == "十" and t[1] in _CN_NUM_MAP:
        return 10 + _CN_NUM_MAP[t[1]]
    if len(t) == 2 and t[1] == "十" and t[0] in _CN_NUM_MAP:
        return _CN_NUM_MAP[t[0]] * 10
    if len(t) == 3 and t[1] == "十" and t[0] in _CN_NUM_MAP and t[2] in _CN_NUM_MAP:
        return _CN_NUM_MAP[t[0]] * 10 + _CN_NUM_MAP[t[2]]
    return None


def _extract_result_item_index(question: str) -> int | None:
    compact = re.sub(r"\s+", "", (question or "").strip())
    compact = compact.strip("，。！？；:!?;:[]【】（）\"'`")
    if not compact:
        return None

    # Avoid treating bare numeric replies as direct indexed reference here.
    if re.fullmatch(r"[0-9一二两三四五六七八九十]+", compact):
        return None

    m = re.fullmatch(r"(?:帮我|请|麻烦)?(?:看下|看看|查看|打开|读下|读一个|查下|查一个|看|查)?第?([0-9一二两三四五六七八九十]+)(?:条|项|个)?(?:证据|内容)?", compact)
    if not m:
        return None
    return _parse_cn_number(m.group(1))


def _extract_file_candidate(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    m = _FILE_PATH_PATTERN.search(raw)
    if not m:
        return ""
    candidate = m.group(1).strip()
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate.strip("“”\"'[]【】（）(),，。；;！？?")


def _extract_numbered_result_items(answer_text: str) -> list[dict]:
    entries: list[dict] = []
    current: dict | None = None
    for raw in (answer_text or "").splitlines():
        line = (raw or "").strip()
        if not line:
            continue

        m = _NUMBERED_RESULT_LINE_PATTERN.match(line)
        if m:
            if current is not None:
                entries.append(current)
            current = {
                "index": int(m.group(1)),
                "title": (m.group(2) or "").strip(),
                "source_raw": "",
            }
            continue

        if current is None:
            continue

        m = _RESULT_SOURCE_LINE_PATTERN.match(line)
        if m and not current["source_raw"]:
            current["source_raw"] = (m.group(1) or "").strip()

    if current is not None:
        entries.append(current)
    return entries


def resolve_result_item_reference(
    *,
    question: str,
    last_answer: str,
    repo_paths: list[str],
) -> dict | None:
    idx = _extract_result_item_index(question)
    if idx is None:
        return None

    entries = _extract_numbered_result_items(last_answer)
    if not entries:
        return {
            "index": idx,
            "entries": [],
            "error": "上一条回答里没有可索引的编号条目。",
            "target": None,
            "source_rel": None,
            "source_items": [],
        }

    target = next((item for item in entries if int(item.get("index", -1)) == idx), None)
    if target is None and 1 <= idx <= len(entries):
        target = entries[idx - 1]

    source_items: list[str] = []
    seen: set[str] = set()
    for item in entries:
        source_raw = str(item.get("source_raw") or "")
        candidate = _extract_file_candidate(source_raw) or _extract_file_candidate(str(item.get("title") or ""))
        if not candidate:
            continue
        matched = find_repo_path_by_reference(candidate, repo_paths)
        if not matched:
            continue
        key = matched.lower().replace(" ", "")
        if key in seen:
            continue
        seen.add(key)
        source_items.append(matched)

    if target is None:
        return {
            "index": idx,
            "entries": entries,
            "error": f"序号超出范围。上一条共 {len(entries)} 条。",
            "target": None,
            "source_rel": None,
            "source_items": source_items,
        }

    source_candidate = _extract_file_candidate(str(target.get("source_raw") or ""))
    if not source_candidate:
        source_candidate = _extract_file_candidate(str(target.get("title") or ""))
    source_rel = find_repo_path_by_reference(source_candidate, repo_paths) if source_candidate else None

    return {
        "index": idx,
        "entries": entries,
        "error": None,
        "target": target,
        "source_rel": source_rel,
        "source_items": source_items,
    }


def format_change_history_answer(events: list[dict]) -> str:
    if not events:
        return "当前还没有文件变更记录。"

    action_map = {
        "rename": "重命名",
        "delete": "删除(软删除)",
    }
    lines = [f"最近 {len(events)} 条文件变更记录（按时间倒序）："]
    for i, evt in enumerate(events, start=1):
        created_at = str(evt.get("created_at", "")).replace("T", " ").replace("+00:00", "Z")
        event_type = action_map.get(evt.get("event_type", ""), evt.get("event_type", "unknown"))
        lines.append(
            f"{i}. [{evt['event_id']}] {event_type}: "
            f"{evt['before_path']} -> {evt['after_path']} "
            f"(SHA {evt['before_sha256'][:8]} -> {evt['after_sha256'][:8]} | {created_at})"
        )
    return "\n".join(lines)


def format_rename_history_answer(events: list[dict]) -> str:
    if not events:
        return "当前还没有重命名记录。"

    lines = ["最近重命名记录如下（按时间倒序）："]
    for i, evt in enumerate(events, start=1):
        created_at = str(evt.get("created_at", "")).replace("T", " ").replace("+00:00", "Z")
        lines.append(
            f"{i}. [{evt['event_id']}] {evt['before_path']} -> {evt['after_path']} "
            f"(SHA {evt['before_sha256'][:8]} -> {evt['after_sha256'][:8]} | {created_at})"
        )
    return "\n".join(lines)
