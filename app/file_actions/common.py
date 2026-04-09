from __future__ import annotations

from pathlib import Path

from app.chat_state_helpers import (
    append_memory,
    print_answer,
    update_state_after_local_answer,
)
from app.dialog.state_machine import ConversationState


def clear_pending_action_state(state: ConversationState) -> None:
    state.pending_action_type = None
    state.pending_action_source_path = None
    state.pending_action_target_path = None
    state.pending_action_requested_text = None
    state.pending_action_preview = None
    state.pending_action_payload = None


def find_repo_path_by_reference(source_ref: str | None, repo_paths: list[str]) -> str | None:
    if not source_ref:
        return None

    ref = source_ref.strip().replace("\\", "/")
    if not ref:
        return None

    ref_norm = ref.lower().replace(" ", "")
    stripped_prefixes = ("这个", "那个", "那", "该", "这", "把")
    for pfx in stripped_prefixes:
        if ref_norm.startswith(pfx):
            ref_norm = ref_norm[len(pfx):]

    for path in repo_paths:
        if path.lower().replace(" ", "") == ref_norm:
            return path

    for path in repo_paths:
        norm_path = path.lower().replace(" ", "")
        if norm_path.endswith(ref_norm):
            return path
        if ref_norm.endswith(norm_path):
            return path

    ref_name = Path(ref).name.lower().replace(" ", "")
    for path in repo_paths:
        if Path(path).name.lower().replace(" ", "") == ref_name:
            return path

    return None


def reply_file_action(
    *,
    state: ConversationState,
    memory_buffer: list[str],
    question: str,
    answer: str,
    start_qa: float,
    local_topic: str,
    is_content_answer: bool,
) -> ConversationState:
    print_answer(answer, start_qa)
    append_memory(memory_buffer, question, answer)
    return update_state_after_local_answer(
        state,
        question=question,
        answer=answer,
        route="file_action",
        local_topic=local_topic,
        is_content_answer=is_content_answer,
    )
