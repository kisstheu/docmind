from __future__ import annotations

from pathlib import Path

from app.dialog.state_machine import ConversationState
from app.file_actions.common import clear_pending_action_state, reply_file_action
from app.file_flows.delete import parse_delete_confirmation_decision
from app.file_flows.rename import parse_confirmation_decision
from app.repo_state_mutations import apply_repo_state_delete, apply_repo_state_rename
from infra.file_change_store import FileChangeStore, collect_file_snapshot


def handle_pending_file_action(
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
    if state.pending_action_type == "rename":
        decision = parse_confirmation_decision(question)
        if decision is None:
            remind = (
                "当前有待确认的重命名操作。\n"
                f"{state.pending_action_preview or ''}\n"
                "请回复“确认重命名”执行，或回复“取消”。"
            )
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=remind,
                start_qa=start_qa,
                local_topic="rename_pending",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        if decision == "cancel":
            canceled = "已取消本次重命名操作。"
            clear_pending_action_state(state)
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=canceled,
                start_qa=start_qa,
                local_topic="rename_canceled",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        source_rel = state.pending_action_source_path
        target_rel = state.pending_action_target_path
        requested_text = state.pending_action_requested_text or ""
        if not source_rel or not target_rel:
            failed = "待执行的重命名状态不完整，已取消。请重新发起改名请求。"
            clear_pending_action_state(state)
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=failed,
                start_qa=start_qa,
                local_topic="rename_failed",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        source_abs = (notes_dir / source_rel).resolve()
        target_abs = (notes_dir / target_rel).resolve()

        if not source_abs.exists():
            failed = f"原文件不存在或已被移动：{source_rel}。已取消本次重命名。"
            clear_pending_action_state(state)
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=failed,
                start_qa=start_qa,
                local_topic="rename_failed",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        if target_abs.exists() and target_abs != source_abs:
            failed = f"目标文件已存在：{target_rel}。请换一个名字后重试。"
            clear_pending_action_state(state)
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=failed,
                start_qa=start_qa,
                local_topic="rename_failed",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        before = collect_file_snapshot(source_abs, notes_dir)
        source_abs.rename(target_abs)
        after = collect_file_snapshot(target_abs, notes_dir)
        event_id = change_store.record_rename(
            notes_dir=notes_dir,
            before=before,
            after=after,
            reason="user_confirmed_rename",
            requested_text=requested_text,
            confirmed_text=question,
        )

        apply_repo_state_rename(
            repo_state,
            notes_dir=notes_dir,
            old_rel_path=source_rel,
            new_rel_path=target_rel,
        )
        if current_focus_file == source_rel:
            current_focus_file = target_rel
        if state.last_result_set_items:
            state.last_result_set_items = [
                target_rel if x == source_rel else x
                for x in state.last_result_set_items
            ]

        clear_pending_action_state(state)
        success = (
            "重命名完成。\n"
            f"- 原路径: {source_rel}\n"
            f"- 新路径: {target_rel}\n"
            f"- 变更记录ID: {event_id}\n"
            f"- 原始SHA256: {before['sha256'][:16]}...\n"
            f"- 当前SHA256: {after['sha256'][:16]}..."
        )
        state = reply_file_action(
            state=state,
            memory_buffer=memory_buffer,
            question=question,
            answer=success,
            start_qa=start_qa,
            local_topic="rename_done",
            is_content_answer=True,
        )
        return True, state, current_focus_file

    if state.pending_action_type == "delete":
        decision = parse_delete_confirmation_decision(question)
        if decision is None:
            remind = (
                "当前有待确认的删除操作。\n"
                f"{state.pending_action_preview or ''}\n"
                "请回复“确认删除”执行，或回复“取消”。"
            )
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=remind,
                start_qa=start_qa,
                local_topic="delete_pending",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        if decision == "cancel":
            canceled = "已取消本次删除操作。"
            clear_pending_action_state(state)
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=canceled,
                start_qa=start_qa,
                local_topic="delete_canceled",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        source_rel = state.pending_action_source_path
        target_rel = state.pending_action_target_path
        requested_text = state.pending_action_requested_text or ""
        if not source_rel or not target_rel:
            failed = "待执行的删除状态不完整，已取消。请重新发起删除请求。"
            clear_pending_action_state(state)
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=failed,
                start_qa=start_qa,
                local_topic="delete_failed",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        source_abs = (notes_dir / source_rel).resolve()
        target_abs = (notes_dir / target_rel).resolve()

        if not source_abs.exists() or not source_abs.is_file():
            failed = f"原文件不存在或已被移动：{source_rel}。已取消本次删除。"
            clear_pending_action_state(state)
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=failed,
                start_qa=start_qa,
                local_topic="delete_failed",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        target_abs.parent.mkdir(parents=True, exist_ok=True)
        before = collect_file_snapshot(source_abs, notes_dir)
        source_abs.rename(target_abs)
        after = collect_file_snapshot(target_abs, notes_dir)
        event_id = change_store.record_delete(
            notes_dir=notes_dir,
            before=before,
            after=after,
            reason="user_confirmed_soft_delete",
            requested_text=requested_text,
            confirmed_text=question,
        )

        apply_repo_state_delete(
            repo_state,
            notes_dir=notes_dir,
            old_rel_path=source_rel,
        )
        if current_focus_file == source_rel:
            current_focus_file = None
        if state.last_result_set_items:
            state.last_result_set_items = [
                x for x in state.last_result_set_items if x != source_rel
            ]
            if not state.last_result_set_items:
                state.last_result_set_entity_type = None

        clear_pending_action_state(state)
        success = (
            "删除完成（已移动到回收站）。\n"
            f"- 原路径: {source_rel}\n"
            f"- 回收站路径: {target_rel}\n"
            f"- 变更记录ID: {event_id}\n"
            f"- 文件SHA256: {after['sha256'][:16]}..."
        )
        state = reply_file_action(
            state=state,
            memory_buffer=memory_buffer,
            question=question,
            answer=success,
            start_qa=start_qa,
            local_topic="delete_done",
            is_content_answer=True,
        )
        return True, state, current_focus_file

    return False, state, current_focus_file
