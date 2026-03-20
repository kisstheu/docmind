from __future__ import annotations

import time


def clean_answer_for_memory(text: str) -> str:
    return (text or "").replace("\n", " ").replace("*", "").replace("#", "")


def append_memory(memory_buffer: list[str], question: str, answer: str) -> None:
    clean_reply = clean_answer_for_memory(answer)
    memory_buffer.extend([
        f"用户：{question}",
        f"AI：{clean_reply[:150] + ('...' if len(clean_reply) > 150 else '')}",
    ])


def print_answer(answer: str, start_qa: float) -> None:
    print(f"\nAI回答：\n{answer}")
    print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")


def update_state_after_local_answer(
    conversation_state,
    *,
    question: str,
    answer: str,
    route: str,
    local_topic: str | None,
    is_content_answer: bool,
):
    clean_reply = clean_answer_for_memory(answer)

    conversation_state.last_user_question = question
    conversation_state.last_route = route
    conversation_state.last_local_topic = local_topic
    conversation_state.last_answer_preview = clean_reply[:200]

    if route == "smalltalk":
        conversation_state.mode = "smalltalk"
    elif route == "repo_meta":
        conversation_state.mode = "repo_meta"
    else:
        conversation_state.mode = "content"

    if is_content_answer:
        conversation_state.last_content_user_question = question
        conversation_state.last_content_route = route
        conversation_state.last_content_topic = local_topic

    return conversation_state


def update_state_after_retrieval_answer(conversation_state, question: str, answer: str):
    clean_reply = clean_answer_for_memory(answer)

    conversation_state.mode = "content"
    conversation_state.last_user_question = question
    conversation_state.last_route = "normal_retrieval"
    conversation_state.last_local_topic = None
    conversation_state.last_answer_preview = clean_reply[:200]
    conversation_state.last_content_user_question = question
    conversation_state.last_content_route = "normal_retrieval"
    conversation_state.last_content_topic = None

    return conversation_state