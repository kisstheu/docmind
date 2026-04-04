from __future__ import annotations

from pathlib import Path

from app.dialog.state_machine import ConversationState
from app.file_actions.common import find_repo_path_by_reference, reply_file_action
from app.file_flows.delete import build_delete_preview, is_delete_request, resolve_delete_source_file
from app.file_flows.rename import (
    build_rename_preview,
    extract_new_name_candidate,
    is_rename_request,
    normalize_target_filename,
    resolve_source_file,
)


def handle_rename_request_action(
    *,
    question: str,
    start_qa: float,
    state: ConversationState,
    memory_buffer: list[str],
    current_focus_file: str | None,
    repo_paths: list[str],
    notes_dir: Path,
) -> tuple[bool, ConversationState, str | None]:
    if not is_rename_request(question):
        return False, state, current_focus_file

    source_hint = resolve_source_file(
        question=question,
        current_focus_file=current_focus_file,
        last_result_set_items=state.last_result_set_items,
    )
    source_rel = find_repo_path_by_reference(source_hint, repo_paths)
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
        tip = f"我识别到目标文件 `{source_rel}`，但没识别到新名称。请明确说“改成 xxx”。"
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


def handle_delete_request_action(
    *,
    question: str,
    start_qa: float,
    state: ConversationState,
    memory_buffer: list[str],
    current_focus_file: str | None,
    repo_paths: list[str],
    notes_dir: Path,
) -> tuple[bool, ConversationState, str | None]:
    if not is_delete_request(question):
        return False, state, current_focus_file

    source_hint = resolve_delete_source_file(
        question=question,
        current_focus_file=current_focus_file,
        last_result_set_items=state.last_result_set_items,
    )
    source_rel = find_repo_path_by_reference(source_hint, repo_paths)
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
