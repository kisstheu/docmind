from __future__ import annotations

from pathlib import Path

from app.chat_state_helpers import extract_file_items
from app.dialog.state_machine import ConversationState
from app.file_actions.common import reply_file_action
from app.file_actions.request_mutations import (
    handle_delete_request_action,
    handle_organize_request_action,
    handle_rename_request_action,
)
from app.file_actions.request_resolution import (
    format_change_history_answer,
    format_rename_history_answer,
    resolve_result_item_reference as _resolve_result_item_reference,
)
from app.file_flows.history import extract_history_limit, is_change_history_query
from app.file_flows.image_view import (
    create_shadow_image_copy,
    is_image_view_index_selection_request,
    is_image_view_request,
    open_image_with_system_viewer,
    resolve_image_from_result_set,
)
from app.file_flows.rename import is_rename_history_query
from infra.file_change_store import FileChangeStore

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _handle_result_item_reference(
    *,
    question: str,
    start_qa: float,
    state: ConversationState,
    memory_buffer: list[str],
    current_focus_file: str | None,
    repo_paths: list[str],
) -> tuple[bool, ConversationState, str | None]:
    last_answer = (state.last_answer_text or state.last_answer_preview or "").strip()
    item_ref = _resolve_result_item_reference(
        question=question,
        last_answer=last_answer,
        repo_paths=repo_paths,
    )
    if item_ref is None:
        return False, state, current_focus_file

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
        return False, state, current_focus_file

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


def _handle_result_item_image_open(
    *,
    question: str,
    start_qa: float,
    state: ConversationState,
    memory_buffer: list[str],
    current_focus_file: str | None,
    item_ref: dict,
    notes_dir: Path,
) -> tuple[bool, ConversationState, str | None]:
    target = item_ref.get("target") or {}
    target_index = int(target.get("index", item_ref.get("index") or 0))
    target_title = (str(target.get("title") or "") or "（无标题）").strip()
    target_rel = (item_ref.get("source_rel") or "").strip()
    if not target_rel:
        return False, state, current_focus_file

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


def _handle_image_view(
    *,
    question: str,
    start_qa: float,
    state: ConversationState,
    memory_buffer: list[str],
    current_focus_file: str | None,
    repo_paths: list[str],
    notes_dir: Path,
) -> tuple[bool, ConversationState, str | None]:
    should_handle_image_view = is_image_view_request(question)
    if (
        not should_handle_image_view
        and (state.last_local_topic or "") in {"image_view_pending", "image_view_done"}
        and is_image_view_index_selection_request(question)
    ):
        should_handle_image_view = True

    if not should_handle_image_view:
        return False, state, current_focus_file

    last_answer = (state.last_answer_text or state.last_answer_preview or "").strip()
    preferred_rel: str | None = None
    if last_answer:
        prev_turn_files = extract_file_items(last_answer)
        if prev_turn_files:
            preferred_rel = prev_turn_files[-1]

    target_rel, tip = resolve_image_from_result_set(
        question=question,
        last_result_set_items=state.last_result_set_items,
        current_focus_file=current_focus_file,
        repo_paths=repo_paths,
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
    model_emb=None,
) -> tuple[bool, ConversationState, str | None]:
    repo_paths = list(repo_state.paths)

    if is_change_history_query(question):
        limit = extract_history_limit(question, default=20, max_limit=100)
        events = change_store.list_recent_events(notes_dir=notes_dir, limit=limit)
        history_answer = format_change_history_answer(events)
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
        history_answer = format_rename_history_answer(recent_events)
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
        repo_paths=repo_paths,
    )
    if item_ref is not None:
        source_items = list(item_ref.get("source_items") or [])
        if source_items:
            state.last_result_set_items = source_items
            state.last_result_set_entity_type = "文件"

        target_rel = (item_ref.get("source_rel") or "").strip()
        if target_rel and Path(target_rel).suffix.lower() in _IMAGE_EXTENSIONS:
            return _handle_result_item_image_open(
                question=question,
                start_qa=start_qa,
                state=state,
                memory_buffer=memory_buffer,
                current_focus_file=current_focus_file,
                item_ref=item_ref,
                notes_dir=notes_dir,
            )
        return _handle_result_item_reference(
            question=question,
            start_qa=start_qa,
            state=state,
            memory_buffer=memory_buffer,
            current_focus_file=current_focus_file,
            repo_paths=repo_paths,
        )

    handled, state, current_focus_file = _handle_image_view(
        question=question,
        start_qa=start_qa,
        state=state,
        memory_buffer=memory_buffer,
        current_focus_file=current_focus_file,
        repo_paths=repo_paths,
        notes_dir=notes_dir,
    )
    if handled:
        return True, state, current_focus_file

    handled, state, current_focus_file = handle_organize_request_action(
        question=question,
        start_qa=start_qa,
        state=state,
        memory_buffer=memory_buffer,
        current_focus_file=current_focus_file,
        repo_state=repo_state,
        model_emb=model_emb,
        repo_paths=repo_paths,
        notes_dir=notes_dir,
    )
    if handled:
        return True, state, current_focus_file

    handled, state, current_focus_file = handle_rename_request_action(
        question=question,
        start_qa=start_qa,
        state=state,
        memory_buffer=memory_buffer,
        current_focus_file=current_focus_file,
        repo_paths=repo_paths,
        notes_dir=notes_dir,
    )
    if handled:
        return True, state, current_focus_file

    handled, state, current_focus_file = handle_delete_request_action(
        question=question,
        start_qa=start_qa,
        state=state,
        memory_buffer=memory_buffer,
        current_focus_file=current_focus_file,
        repo_paths=repo_paths,
        notes_dir=notes_dir,
    )
    if handled:
        return True, state, current_focus_file

    return False, state, current_focus_file
