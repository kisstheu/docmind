from __future__ import annotations

from pathlib import Path

from app.dialog.state_machine import ConversationState
from app.file_actions.pending_delete import handle_pending_delete_action
from app.file_actions.pending_organize import handle_pending_organize_action
from app.file_actions.pending_paths import replace_result_set_paths as _replace_result_set_paths
from app.file_actions.pending_rename import handle_pending_rename_action
from infra.file_change_store import FileChangeStore


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
        return handle_pending_rename_action(
            question=question,
            start_qa=start_qa,
            state=state,
            memory_buffer=memory_buffer,
            current_focus_file=current_focus_file,
            repo_state=repo_state,
            model_emb=model_emb,
            notes_dir=notes_dir,
            change_store=change_store,
        )
    if state.pending_action_type == "organize":
        return handle_pending_organize_action(
            question=question,
            start_qa=start_qa,
            state=state,
            memory_buffer=memory_buffer,
            current_focus_file=current_focus_file,
            repo_state=repo_state,
            model_emb=model_emb,
            notes_dir=notes_dir,
            change_store=change_store,
        )
    if state.pending_action_type == "delete":
        return handle_pending_delete_action(
            question=question,
            start_qa=start_qa,
            state=state,
            memory_buffer=memory_buffer,
            current_focus_file=current_focus_file,
            repo_state=repo_state,
            model_emb=model_emb,
            notes_dir=notes_dir,
            change_store=change_store,
        )
    return False, state, current_focus_file
