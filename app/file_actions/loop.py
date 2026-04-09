from __future__ import annotations

from pathlib import Path

from app.dialog.state_machine import ConversationState
from app.file_actions.pending import handle_pending_file_action
from app.file_actions.requests import handle_requested_file_action
from infra.file_change_store import FileChangeStore


def handle_file_action_turn(
    *,
    question: str,
    start_qa: float,
    state: ConversationState,
    memory_buffer: list[str],
    current_focus_file: str | None,
    repo_state,
    model_emb,
    notes_dir: Path,
    change_store: FileChangeStore,
) -> tuple[bool, ConversationState, str | None]:
    handled, state, current_focus_file = handle_pending_file_action(
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
    if handled:
        return True, state, current_focus_file

    return handle_requested_file_action(
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
