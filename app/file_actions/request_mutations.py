from __future__ import annotations

import json
from pathlib import Path

from ai.repo_meta.category import build_local_category_assignment_map
from app.dialog.state_machine import ConversationState
from app.file_actions.common import find_repo_path_by_reference, reply_file_action
from app.file_flows.delete import build_delete_preview, is_delete_request, resolve_delete_source_file
from app.file_flows.organize import (
    build_category_organize_preview,
    is_brief_organize_followup_request,
    is_organize_request,
    resolve_organize_source_paths,
)
from app.file_flows.rename import (
    build_rename_preview,
    extract_new_name_candidate,
    is_rename_request,
    normalize_target_filename,
    resolve_source_file,
)


def handle_organize_request_action(
    *,
    question: str,
    start_qa: float,
    state: ConversationState,
    memory_buffer: list[str],
    current_focus_file: str | None,
    repo_state,
    model_emb,
    repo_paths: list[str],
    notes_dir: Path,
) -> tuple[bool, ConversationState, str | None]:
    is_contextual_followup = (
        is_brief_organize_followup_request(question)
        and bool(state.last_result_set_items)
        and state.last_result_set_entity_type == "文件"
        and bool(state.last_category_context_answer)
    )
    if not is_organize_request(question) and not is_contextual_followup:
        return False, state, current_focus_file

    previous_summary = state.last_category_context_answer
    if not previous_summary:
        tip = "我还没有可直接复用的当前分类结果。你可以先问“再概括一下”或“可以列出每类的数量吗？”。"
        state = reply_file_action(
            state=state,
            memory_buffer=memory_buffer,
            question=question,
            answer=tip,
            start_qa=start_qa,
            local_topic="organize_pending",
            is_content_answer=False,
        )
        return True, state, current_focus_file

    _, path_category_map = build_local_category_assignment_map(
        repo_state,
        previous_summary=previous_summary,
        model_emb=model_emb,
    )
    if not path_category_map:
        tip = "我拿到了分类标题，但当前还缺少可落地的本地文件归类映射，暂时没法直接整理。"
        state = reply_file_action(
            state=state,
            memory_buffer=memory_buffer,
            question=question,
            answer=tip,
            start_qa=start_qa,
            local_topic="organize_failed",
            is_content_answer=False,
        )
        return True, state, current_focus_file

    source_paths, scope = resolve_organize_source_paths(
        question=question,
        last_result_set_items=state.last_result_set_items,
        repo_paths=repo_paths,
    )
    if not source_paths:
        if scope == "remaining":
            tip = "其余未整理文件里，暂时没有可继续按当前分类整理的对象了。"
        else:
            tip = "当前没有可用于分类整理的文件范围。你可以先列出一批文件，或明确说“整理所有文件”。"
        state = reply_file_action(
            state=state,
            memory_buffer=memory_buffer,
            question=question,
            answer=tip,
            start_qa=start_qa,
            local_topic="organize_pending",
            is_content_answer=False,
        )
        return True, state, current_focus_file

    preview_text, payload = build_category_organize_preview(
        notes_dir=notes_dir,
        source_paths=source_paths,
        path_category_map=path_category_map,
        scope=scope,
    )
    if not preview_text or not payload:
        tip = "这批文件目前没法按当前分类生成整理计划，可能是文件不存在，或还没有对应的本地分类结果。"
        state = reply_file_action(
            state=state,
            memory_buffer=memory_buffer,
            question=question,
            answer=tip,
            start_qa=start_qa,
            local_topic="organize_failed",
            is_content_answer=False,
        )
        return True, state, current_focus_file

    state.pending_action_type = "organize"
    state.pending_action_source_path = None
    state.pending_action_target_path = payload.get("root_rel_path")
    state.pending_action_requested_text = question
    state.pending_action_preview = preview_text
    state.pending_action_payload = json.dumps(payload, ensure_ascii=False)

    state = reply_file_action(
        state=state,
        memory_buffer=memory_buffer,
        question=question,
        answer=preview_text,
        start_qa=start_qa,
        local_topic="organize_preview",
        is_content_answer=False,
    )
    return True, state, current_focus_file


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
