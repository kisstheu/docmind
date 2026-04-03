from __future__ import annotations

import re
from pathlib import Path

from app.dialog.state_machine import ConversationState
from app.file_actions.common import find_repo_path_by_reference, reply_file_action
from app.file_change_history_flow import extract_history_limit, is_change_history_query
from app.file_delete_flow import (
    build_delete_preview,
    is_delete_request,
    resolve_delete_source_file,
)
from app.file_image_view_flow import (
    create_shadow_image_copy,
    is_image_view_index_selection_request,
    is_image_view_request,
    open_image_with_system_viewer,
    resolve_image_from_result_set,
)
from app.file_rename_flow import (
    build_rename_preview,
    extract_new_name_candidate,
    is_rename_history_query,
    is_rename_request,
    normalize_target_filename,
    resolve_source_file,
)
from app.chat_state_helpers import extract_file_items
from infra.file_change_store import FileChangeStore

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_RESULT_ITEM_INDEX_QUERY_PATTERN = re.compile(
    r"^(?:帮我|请|麻烦(?:你)?|劳烦你)?"
    r"(?:看下|看看|查看|打开|读下|读一下|查下|查一下|看|查)?"
    r"(?:第)?([0-9一二两三四五六七八九十]+)"
    r"(?:条|项|个)?(?:条证据|个证据|条内容|个内容)?(?:吧|呀|呢|吗|嘛)?$"
)
_NUMBERED_RESULT_LINE_PATTERN = re.compile(r"^\s*(\d{1,3})[.、]\s*(.+?)\s*$")
_RESULT_SOURCE_LINE_PATTERN = re.compile(
    r"^\s*(?:[-*•]\s*)?(?:来源|出处|原文件|源文件)\s*[：:]\s*(.+?)\s*$",
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
    compact = compact.strip("，。！？；：,.!?;:[]【】()（）\"'`")
    if not compact:
        return None
    if re.fullmatch(r"[0-9一二两三四五六七八九十]+", compact):
        return None
    m = _RESULT_ITEM_INDEX_QUERY_PATTERN.fullmatch(compact)
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
    return candidate.strip("“”\"'[]【】（）()，,。；;：:")


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


def _resolve_result_item_reference(
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
            "error": f"序号超出范围。上一条共有 {len(entries)} 条。",
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


def handle_requested_file_action(
    *,
    question: str,
    start_qa: float,
    state: ConversationState,
    memory_buffer: list[str],
    current_focus_file: str | None,
    repo_state,
    notes_dir: Path,
    change_store: FileChangeStore,
) -> tuple[bool, ConversationState, str | None]:
    if is_change_history_query(question):
        limit = extract_history_limit(question, default=20, max_limit=100)
        events = change_store.list_recent_events(notes_dir=notes_dir, limit=limit)
        if not events:
            history_answer = "当前还没有文件变更记录。"
        else:
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
            history_answer = "\n".join(lines)

        state = reply_file_action(
            state=state,
            memory_buffer=memory_buffer,
            question=question,
            answer=history_answer,
            start_qa=start_qa,
            local_topic="change_history",
            is_content_answer=False,
        )
        return True, state, current_focus_file

    if is_rename_history_query(question):
        recent_events = change_store.list_recent_renames(notes_dir=notes_dir, limit=20)
        if not recent_events:
            history_answer = "当前还没有重命名记录。"
        else:
            lines = ["最近重命名记录如下（按时间倒序）："]
            for i, evt in enumerate(recent_events, start=1):
                created_at = str(evt.get("created_at", "")).replace("T", " ").replace("+00:00", "Z")
                lines.append(
                    f"{i}. [{evt['event_id']}] {evt['before_path']} -> {evt['after_path']} "
                    f"(SHA {evt['before_sha256'][:8]} -> {evt['after_sha256'][:8]} | {created_at})"
                )
            history_answer = "\n".join(lines)

        state = reply_file_action(
            state=state,
            memory_buffer=memory_buffer,
            question=question,
            answer=history_answer,
            start_qa=start_qa,
            local_topic="rename_history",
            is_content_answer=False,
        )
        return True, state, current_focus_file

    last_answer = (state.last_answer_text or state.last_answer_preview or "").strip()
    item_ref = _resolve_result_item_reference(
        question=question,
        last_answer=last_answer,
        repo_paths=list(repo_state.paths),
    )
    if item_ref is not None:
        source_items = list(item_ref.get("source_items") or [])
        if source_items:
            state.last_result_set_items = source_items
            state.last_result_set_entity_type = "文件"

        err = (item_ref.get("error") or "").strip()
        if err:
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=err,
                start_qa=start_qa,
                local_topic="result_item_pending",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        target = item_ref.get("target") or {}
        target_index = int(target.get("index", item_ref.get("index") or 0))
        target_title = (str(target.get("title") or "") or "（无标题）").strip()
        target_rel = (item_ref.get("source_rel") or "").strip()
        if target_rel and Path(target_rel).suffix.lower() in _IMAGE_EXTENSIONS:
            shadow_path, shadow_err = create_shadow_image_copy(
                notes_dir=notes_dir,
                source_rel_path=target_rel,
            )
            if not shadow_path:
                state = reply_file_action(
                    state=state,
                    memory_buffer=memory_buffer,
                    question=question,
                    answer=shadow_err or "创建影子查看文件失败。",
                    start_qa=start_qa,
                    local_topic="image_view_failed",
                    is_content_answer=False,
                )
                return True, state, current_focus_file

            opened, open_err = open_image_with_system_viewer(shadow_path)
            if not opened:
                answer = (
                    "已生成影子图片，但自动打开失败。\n"
                    f"- 原文件：{target_rel}\n"
                    f"- 影子副本：{shadow_path}\n"
                    f"- 错误：{open_err or 'unknown'}"
                )
                state = reply_file_action(
                    state=state,
                    memory_buffer=memory_buffer,
                    question=question,
                    answer=answer,
                    start_qa=start_qa,
                    local_topic="image_view_failed",
                    is_content_answer=False,
                )
                return True, state, current_focus_file

            answer = (
                "已按编号定位并打开影子图片（原文件不会被改动）：\n"
                f"- 条目：第{target_index}项 {target_title}\n"
                f"- 原文件：{target_rel}\n"
                f"- 影子副本：{shadow_path}\n"
                "可继续说“打开第2张图”切换查看。"
            )
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=answer,
                start_qa=start_qa,
                local_topic="image_view_done",
                is_content_answer=False,
            )
            return True, state, target_rel

        lines = [f"已定位上一条第{target_index}项：{target_title}"]
        if target_rel:
            lines.append(f"- 来源文件：{target_rel}")
            lines.append("可继续说“查看图片”来打开原图（如果该文件是图片）。")
            next_focus = target_rel
        else:
            source_raw = (str(target.get("source_raw") or "") or "").strip()
            if source_raw:
                lines.append(f"- 来源线索：{source_raw}")
            else:
                lines.append("- 该条目未包含可解析的来源文件路径。")
            next_focus = current_focus_file

        state = reply_file_action(
            state=state,
            memory_buffer=memory_buffer,
            question=question,
            answer="\n".join(lines),
            start_qa=start_qa,
            local_topic="result_item_located",
            is_content_answer=False,
        )
        return True, state, next_focus

    should_handle_image_view = is_image_view_request(question)
    if (
        not should_handle_image_view
        and (state.last_local_topic or "") in {"image_view_pending", "image_view_done"}
        and is_image_view_index_selection_request(question)
    ):
        should_handle_image_view = True

    if should_handle_image_view:
        preferred_rel: str | None = None
        if last_answer:
            prev_turn_files = extract_file_items(last_answer)
            if prev_turn_files:
                preferred_rel = prev_turn_files[-1]

        target_rel, tip = resolve_image_from_result_set(
            question=question,
            last_result_set_items=state.last_result_set_items,
            current_focus_file=current_focus_file,
            repo_paths=list(repo_state.paths),
            preferred_rel_path=preferred_rel,
        )
        if not target_rel:
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=tip or "没有定位到要查看的图片。",
                start_qa=start_qa,
                local_topic="image_view_pending",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        shadow_path, shadow_err = create_shadow_image_copy(
            notes_dir=notes_dir,
            source_rel_path=target_rel,
        )
        if not shadow_path:
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=shadow_err or "创建影子查看文件失败。",
                start_qa=start_qa,
                local_topic="image_view_failed",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        opened, open_err = open_image_with_system_viewer(shadow_path)
        if not opened:
            answer = (
                "已生成影子图片，但自动打开失败。\n"
                f"- 原文件：{target_rel}\n"
                f"- 影子副本：{shadow_path}\n"
                f"- 错误：{open_err or 'unknown'}"
            )
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=answer,
                start_qa=start_qa,
                local_topic="image_view_failed",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        answer = (
            "已打开影子图片（原文件不会被改动）：\n"
            f"- 原文件：{target_rel}\n"
            f"- 影子副本：{shadow_path}\n"
            "可继续说“打开第2张图”切换查看。"
        )
        state = reply_file_action(
            state=state,
            memory_buffer=memory_buffer,
            question=question,
            answer=answer,
            start_qa=start_qa,
            local_topic="image_view_done",
            is_content_answer=False,
        )
        return True, state, target_rel

    if is_rename_request(question):
        source_hint = resolve_source_file(
            question=question,
            current_focus_file=current_focus_file,
            last_result_set_items=state.last_result_set_items,
        )
        source_rel = find_repo_path_by_reference(source_hint, list(repo_state.paths))
        new_name_candidate = extract_new_name_candidate(question)

        if not source_rel:
            tip = "我没定位到要改名的文件。请带上完整文件名再说一次。"
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=tip,
                start_qa=start_qa,
                local_topic="rename_pending",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        if not new_name_candidate:
            tip = f"我识别到了目标文件 `{source_rel}`，但没识别到新名称。请明确说“改成 xxx”。"
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=tip,
                start_qa=start_qa,
                local_topic="rename_pending",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        target_rel = normalize_target_filename(new_name_candidate, source_rel)
        if not target_rel:
            tip = "新文件名不合法或格式不支持。请换一个不含特殊字符的名字。"
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=tip,
                start_qa=start_qa,
                local_topic="rename_pending",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        if target_rel == source_rel:
            tip = f"文件名已经是 `{Path(source_rel).name}`，无需修改。"
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=tip,
                start_qa=start_qa,
                local_topic="rename_pending",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        preview_text, payload = build_rename_preview(
            notes_dir=notes_dir,
            source_rel_path=source_rel,
            target_rel_path=target_rel,
        )
        if not preview_text or not payload:
            tip = "重命名预检查失败：源文件不存在，或目标文件已存在。"
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=tip,
                start_qa=start_qa,
                local_topic="rename_pending",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        state.pending_action_type = "rename"
        state.pending_action_source_path = payload["source_rel_path"]
        state.pending_action_target_path = payload["target_rel_path"]
        state.pending_action_requested_text = question
        state.pending_action_preview = preview_text

        state = reply_file_action(
            state=state,
            memory_buffer=memory_buffer,
            question=question,
            answer=preview_text,
            start_qa=start_qa,
            local_topic="rename_preview",
            is_content_answer=False,
        )
        return True, state, current_focus_file

    if is_delete_request(question):
        source_hint = resolve_delete_source_file(
            question=question,
            current_focus_file=current_focus_file,
            last_result_set_items=state.last_result_set_items,
        )
        source_rel = find_repo_path_by_reference(source_hint, list(repo_state.paths))
        if not source_rel:
            tip = "我没定位到要删除的文件，请带上完整文件名再说一次。"
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=tip,
                start_qa=start_qa,
                local_topic="delete_pending",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        preview_text, payload = build_delete_preview(
            notes_dir=notes_dir,
            source_rel_path=source_rel,
        )
        if not preview_text or not payload:
            tip = "删除预检查失败：源文件不存在，或文件类型不支持。"
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=tip,
                start_qa=start_qa,
                local_topic="delete_failed",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        state.pending_action_type = "delete"
        state.pending_action_source_path = payload["source_rel_path"]
        state.pending_action_target_path = payload["target_rel_path"]
        state.pending_action_requested_text = question
        state.pending_action_preview = preview_text

        state = reply_file_action(
            state=state,
            memory_buffer=memory_buffer,
            question=question,
            answer=preview_text,
            start_qa=start_qa,
            local_topic="delete_preview",
            is_content_answer=False,
        )
        return True, state, current_focus_file

    return False, state, current_focus_file
