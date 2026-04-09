from __future__ import annotations

import json
from pathlib import Path

from app.dialog.state_machine import ConversationState
from app.file_actions.common import clear_pending_action_state, reply_file_action
from app.file_flows.delete import parse_delete_confirmation_decision
from app.file_flows.rename import parse_confirmation_decision
from app.repo_state_mutations import apply_repo_state_delete, apply_repo_state_rename
from infra.file_change_store import FileChangeStore, collect_file_snapshot


def _replace_result_set_paths(items: list[str] | None, rename_map: dict[str, str]) -> list[str] | None:
    if not items:
        return items
    return [rename_map.get(item, item) for item in items]


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
    model_emb=None,
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

    if state.pending_action_type == "organize":
        decision = parse_confirmation_decision(question)
        normalized_question = (question or "").strip().lower().replace(" ", "")
        if decision is None and any(term in normalized_question for term in ("确认整理", "执行整理")):
            decision = "confirm"
        if decision is None and "取消" in normalized_question:
            decision = "cancel"
        if decision is None:
            remind = (
                "当前有待确认的分类整理操作。\n"
                f"{state.pending_action_preview or ''}\n"
                "请回复“确认整理”执行，或回复“取消”。"
            )
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=remind,
                start_qa=start_qa,
                local_topic="organize_pending",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        if decision == "cancel":
            canceled = "已取消本次分类整理操作。"
            clear_pending_action_state(state)
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=canceled,
                start_qa=start_qa,
                local_topic="organize_canceled",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        requested_text = state.pending_action_requested_text or ""
        try:
            payload = json.loads(state.pending_action_payload or "{}")
        except Exception:
            payload = {}
        moves = list(payload.get("moves") or [])
        root_rel_path = str(payload.get("root_rel_path") or state.pending_action_target_path or "").strip()
        if not moves or not root_rel_path:
            failed = "待执行的分类整理状态不完整，已取消。请重新发起整理请求。"
            clear_pending_action_state(state)
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=failed,
                start_qa=start_qa,
                local_topic="organize_failed",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        rename_map: dict[str, str] = {}
        for move in moves:
            source_rel = str(move.get("source_rel_path") or "").strip()
            target_rel = str(move.get("target_rel_path") or "").strip()
            if not source_rel or not target_rel:
                failed = "整理计划中存在不完整的文件路径，已取消本次整理。"
                clear_pending_action_state(state)
                state = reply_file_action(
                    state=state,
                    memory_buffer=memory_buffer,
                    question=question,
                    answer=failed,
                    start_qa=start_qa,
                    local_topic="organize_failed",
                    is_content_answer=False,
                )
                return True, state, current_focus_file

            source_abs = (notes_dir / source_rel).resolve()
            target_abs = (notes_dir / target_rel).resolve()
            if not source_abs.exists() or not source_abs.is_file():
                failed = f"原文件不存在或已被移动：{source_rel}。已取消本次整理。"
                clear_pending_action_state(state)
                state = reply_file_action(
                    state=state,
                    memory_buffer=memory_buffer,
                    question=question,
                    answer=failed,
                    start_qa=start_qa,
                    local_topic="organize_failed",
                    is_content_answer=False,
                )
                return True, state, current_focus_file
            if target_abs.exists() and target_abs != source_abs:
                failed = f"整理目标已存在：{target_rel}。已取消本次整理。"
                clear_pending_action_state(state)
                state = reply_file_action(
                    state=state,
                    memory_buffer=memory_buffer,
                    question=question,
                    answer=failed,
                    start_qa=start_qa,
                    local_topic="organize_failed",
                    is_content_answer=False,
                )
                return True, state, current_focus_file

        event_ids: list[int] = []
        category_counts: dict[str, int] = {}
        try:
            for move in moves:
                source_rel = str(move.get("source_rel_path") or "").strip()
                target_rel = str(move.get("target_rel_path") or "").strip()
                category_label = str(move.get("category_label") or "").strip()
                source_abs = (notes_dir / source_rel).resolve()
                target_abs = (notes_dir / target_rel).resolve()
                target_abs.parent.mkdir(parents=True, exist_ok=True)

                before = collect_file_snapshot(source_abs, notes_dir)
                source_abs.rename(target_abs)
                after = collect_file_snapshot(target_abs, notes_dir)
                event_id = change_store.record_rename(
                    notes_dir=notes_dir,
                    before=before,
                    after=after,
                    reason="user_confirmed_organize_by_category",
                    requested_text=requested_text,
                    confirmed_text=question,
                )
                event_ids.append(event_id)
                rename_map[source_rel] = target_rel
                if category_label:
                    category_counts[category_label] = category_counts.get(category_label, 0) + 1

                apply_repo_state_rename(
                    repo_state,
                    notes_dir=notes_dir,
                    old_rel_path=source_rel,
                    new_rel_path=target_rel,
                )
        except Exception as exc:
            failed = f"分类整理执行中断，已完成 {len(event_ids)} 个文件，错误：{exc}"
            clear_pending_action_state(state)
            state = reply_file_action(
                state=state,
                memory_buffer=memory_buffer,
                question=question,
                answer=failed,
                start_qa=start_qa,
                local_topic="organize_failed",
                is_content_answer=False,
            )
            return True, state, current_focus_file

        if current_focus_file in rename_map:
            current_focus_file = rename_map[current_focus_file]
        if state.last_result_set_items:
            state.last_result_set_items = _replace_result_set_paths(state.last_result_set_items, rename_map)

        clear_pending_action_state(state)
        summary_parts = [f"{label} {count} 个" for label, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))]
        success_lines = [
            f"分类整理完成，共移动 {len(rename_map)} 个文件。",
            f"- 目标根目录: {root_rel_path}",
            f"- 变更记录数: {len(event_ids)}",
        ]
        if summary_parts:
            success_lines.append(f"- 分类分布: {'；'.join(summary_parts)}")
        if event_ids:
            success_lines.append(f"- 首个变更记录ID: {event_ids[0]}")
        state = reply_file_action(
            state=state,
            memory_buffer=memory_buffer,
            question=question,
            answer="\n".join(success_lines),
            start_qa=start_qa,
            local_topic="organize_done",
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
