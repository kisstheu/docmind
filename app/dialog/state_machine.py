from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.context_anchor import is_context_dependent_question
from app.dialog.repo_meta_rules import (
    is_entity_lookup_request,
    is_list_format_modifier,
    is_repo_meta_request,
    is_structured_output_request,
    is_system_capability_request,
    looks_like_repo_size_consistency_followup,
    looks_like_repo_time_question,
)
from app.dialog.result_set import (
    build_result_set_followup_query,
    extract_result_set_from_answer,
    last_turn_looks_like_enumeration,
    looks_like_result_set_continuation_followup,
    looks_like_result_set_comparison_followup,
    looks_like_result_set_followup,
)
from app.dialog_utils import (
    is_action_request,
    is_smalltalk_message,
    is_followup_question,
    is_judgment_request,
    is_query_correction,
    is_relationship_analysis_request,
    is_repo_meta_confirmation,
)


@dataclass
class ConversationState:
    mode: str = "idle"
    last_user_question: str | None = None
    last_route: str | None = None
    last_local_topic: str | None = None
    last_answer_preview: str | None = None

    last_content_user_question: str | None = None
    last_content_route: str | None = None
    last_content_topic: str | None = None

    last_effective_search_query: str | None = None
    last_answer_text: str | None = None
    last_answer_type: str | None = None
    last_result_set_query: str | None = None
    last_result_set_items: list[str] | None = None
    last_result_set_entity_type: str | None = None

    pending_action_type: str | None = None
    pending_action_source_path: str | None = None
    pending_action_target_path: str | None = None
    pending_action_requested_text: str | None = None
    pending_action_preview: str | None = None


@dataclass
class DialogEvent:
    name: str
    route_hint: Optional[str] = None
    merged_query: Optional[str] = None


def detect_dialog_event(question: str, state: ConversationState, logger) -> DialogEvent:
    rs_match = looks_like_result_set_followup(question)
    has_state = state is not None
    last_answer_text = state.last_answer_text or state.last_answer_preview or ""
    has_last_answer = bool(last_answer_text)
    enum_like = last_turn_looks_like_enumeration(last_answer_text)
    last_answer_type = state.last_answer_type

    logger.debug(
        f"🧪 [result_set_followup判定] "
        f"rs_match={rs_match} | "
        f"has_state={has_state} | "
        f"has_last_answer={has_last_answer} | "
        f"enum_like={enum_like} | "
        f"last_answer_type={last_answer_type}"
    )
    prev_q = state.last_content_user_question
    prev_route = state.last_content_route
    last_topic = state.last_local_topic

    if state.last_route == "repo_meta" and last_topic == "list_files" and is_list_format_modifier(question):
        return DialogEvent(name="repo_meta_request", route_hint="repo_meta")

    if looks_like_repo_time_question(question, state):
        return DialogEvent(name="repo_meta_request", route_hint="repo_meta")

    if looks_like_repo_size_consistency_followup(question, prev_q):
        return DialogEvent(name="repo_meta_request", route_hint="repo_meta")

    if is_system_capability_request(question):
        return DialogEvent(name="system_capability", route_hint="system_capability")

    if is_repo_meta_request(question):
        return DialogEvent(name="repo_meta_request", route_hint="repo_meta")

    if is_smalltalk_message(question):
        return DialogEvent(name="smalltalk", route_hint="smalltalk")

    # === 1. 纠偏
    if is_query_correction(question) and prev_q:
        return DialogEvent(
            name="query_correction",
            route_hint="normal_retrieval",
            merged_query=f"{prev_q} {question}",
        )

    # === 2. 关系判断
    if is_relationship_analysis_request(question):
        return DialogEvent(name="relationship_analysis", route_hint="normal_retrieval")

    # === 3. 结构化请求
    if is_structured_output_request(question):
        if prev_route == "normal_retrieval" and prev_q:
            return DialogEvent(
                name="structured_request",
                route_hint="normal_retrieval",
                merged_query=f"{prev_q} {question}",
            )
        return DialogEvent(name="structured_request", route_hint="normal_retrieval")

    # === 4. 动作请求
    if is_action_request(question):
        if prev_route == "normal_retrieval" and prev_q and len(question.strip()) <= 12:
            return DialogEvent(
                name="action_request",
                route_hint="normal_retrieval",
                merged_query=f"{prev_q} {question}",
            )
        return DialogEvent(name="action_request", route_hint="normal_retrieval")

    # === 5. 判断请求
    if is_judgment_request(question):
        if prev_route == "normal_retrieval" and prev_q:
            return DialogEvent(
                name="judgment_request",
                route_hint="normal_retrieval",
                merged_query=f"{prev_q} {question}",
            )
        return DialogEvent(name="judgment_request", route_hint="normal_retrieval")

    # === 6. repo_meta 继承
    if prev_route == "repo_meta":
        if is_entity_lookup_request(question):
            return DialogEvent(name="entity_lookup_followup", route_hint="normal_retrieval")

        if is_followup_question(question):
            return DialogEvent(name="repo_followup", route_hint="repo_meta")

        if (
            last_topic in {"category", "category_summary", "category_confirm"}
            and is_repo_meta_confirmation(question)
        ):
            return DialogEvent(name="repo_confirm", route_hint="repo_meta")

        if (
            len(question.strip()) <= 12
            and any(x in question for x in ["粗", "细", "大类", "概括", "方面", "分类"])
        ):
            return DialogEvent(name="repo_followup", route_hint="repo_meta")

    # === 7. 结果集追问（窄继承）
    is_result_set_answer = (
        state.last_answer_type in {"enumeration_company", "enumeration_file", "enumeration_person"}
    ) or enum_like

    if state.last_result_set_items and any(term in question.lower() for term in ["内容", "一样", "相同", "一致"]) and "文件" in question.lower():
        rs_match = True  # 强制设置匹配
    elif state.last_result_set_items and looks_like_result_set_comparison_followup(question):
        rs_match = True

    if prev_route in {"normal_retrieval", "repo_meta"} and rs_match and is_result_set_answer:
        if looks_like_result_set_continuation_followup(question):
            return DialogEvent(name="result_set_expansion_followup", route_hint="normal_retrieval")
        return DialogEvent(name="result_set_followup", route_hint="normal_retrieval")

    # === 8. 内容追问继承
    if prev_route == "normal_retrieval" and is_context_dependent_question(question, state.last_effective_search_query):
        if prev_q:
            return DialogEvent(
                name="content_followup",
                route_hint="normal_retrieval",
                merged_query=f"{prev_q} {question}",
            )
        return DialogEvent(name="content_followup", route_hint="normal_retrieval")

    # === 9. 默认
    return DialogEvent(name="unknown", route_hint=None)


def apply_event_to_state(state: ConversationState, event: DialogEvent) -> ConversationState:
    new_state = ConversationState(
        mode=state.mode,
        last_user_question=state.last_user_question,
        last_route=state.last_route,
        last_local_topic=state.last_local_topic,
        last_answer_preview=state.last_answer_preview,

        last_content_user_question=state.last_content_user_question,
        last_content_route=state.last_content_route,
        last_content_topic=state.last_content_topic,

        last_effective_search_query=state.last_effective_search_query,
        last_answer_text=state.last_answer_text,
        last_answer_type=state.last_answer_type,
        last_result_set_query=state.last_result_set_query,

        last_result_set_items=state.last_result_set_items,
        last_result_set_entity_type=state.last_result_set_entity_type,

        pending_action_type=state.pending_action_type,
        pending_action_source_path=state.pending_action_source_path,
        pending_action_target_path=state.pending_action_target_path,
        pending_action_requested_text=state.pending_action_requested_text,
        pending_action_preview=state.pending_action_preview,
    )

    if event.route_hint == "repo_meta":
        new_state.mode = "repo_meta"
    elif event.route_hint == "normal_retrieval":
        new_state.mode = "content"
    elif event.name == "smalltalk":
        new_state.mode = "smalltalk"

    return new_state


__all__ = [
    "ConversationState",
    "DialogEvent",
    "apply_event_to_state",
    "build_result_set_followup_query",
    "detect_dialog_event",
    "extract_result_set_from_answer",
]
