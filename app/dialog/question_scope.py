from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from app.context_anchor import is_context_dependent_question
from app.dialog.result_set import (
    FileResultSetSelection,
    has_explicit_single_file_result_reference,
    has_selectable_result_set,
    looks_like_result_set_comparison_followup,
    resolve_file_result_set_selection,
)
from app.dialog.task_semantics import (
    is_detail_explanation_request,
    is_document_evaluation_request,
)
from app.dialog_utils import is_content_followup_question, is_summary_followup_request
from app.chat_text.file_lookup import (
    has_explicit_focus_reference,
    looks_like_file_set_content_question,
    looks_like_standalone_general_question,
)


class _QuestionScopeState(Protocol):
    last_result_set_items: list[str] | None
    last_result_set_entity_type: str | None
    last_answer_text: str | None
    last_answer_preview: str | None
    last_result_set_selectable: bool | None
    last_result_set_focus_file: str | None


@dataclass(frozen=True)
class QuestionSignals:
    standalone_general_question: bool
    file_set_content_question: bool
    explicit_focus_reference: bool
    explicit_single_file_result_reference: bool
    result_set_comparison_followup: bool
    content_followup_question: bool
    detail_explanation_request: bool
    summary_followup_request: bool
    context_dependent_question: bool
    document_evaluation_request: bool


@dataclass(frozen=True)
class ScopeDecision:
    clear_current_focus: bool
    effective_focus_file: str | None
    visible_file_paths: tuple[str, ...]
    file_result_set_selection: FileResultSetSelection | None
    selected_file_paths: tuple[str, ...] | None
    selected_result_set_item_turn: bool
    result_set_comparison_turn: bool
    requires_result_set_generation: bool
    result_scope_paths: tuple[str, ...] | None
    query_result_set_items: tuple[str, ...] | None
    query_result_set_entity: str | None
    has_single_focus_scope: bool


def analyze_question_signals(
    question: str,
    *,
    last_effective_search_query: str | None,
) -> QuestionSignals:
    return QuestionSignals(
        standalone_general_question=looks_like_standalone_general_question(question),
        file_set_content_question=looks_like_file_set_content_question(question),
        explicit_focus_reference=has_explicit_focus_reference(question),
        explicit_single_file_result_reference=has_explicit_single_file_result_reference(
            question
        ),
        result_set_comparison_followup=looks_like_result_set_comparison_followup(
            question
        ),
        content_followup_question=is_content_followup_question(question),
        detail_explanation_request=is_detail_explanation_request(question),
        summary_followup_request=is_summary_followup_request(question),
        context_dependent_question=is_context_dependent_question(
            question,
            last_effective_search_query,
        ),
        document_evaluation_request=is_document_evaluation_request(question),
    )


def decide_file_result_set_scope(
    question: str,
    *,
    signals: QuestionSignals,
    state: _QuestionScopeState,
    current_focus_file: str | None,
    event_name: str,
) -> ScopeDecision:
    effective_focus_file = (
        None
        if signals.standalone_general_question
        else current_focus_file or state.last_result_set_focus_file
    )

    visible_file_paths: tuple[str, ...] = ()
    if state.last_result_set_entity_type == "文件" and has_selectable_result_set(
        state.last_result_set_items,
        state.last_result_set_entity_type,
        state.last_answer_text or state.last_answer_preview,
        state.last_result_set_selectable,
    ):
        visible_file_paths = tuple(
            str(path or "").strip()
            for path in (state.last_result_set_items or [])
            if str(path or "").strip()
        )

    file_result_set_selection = resolve_file_result_set_selection(
        question,
        visible_file_paths,
        focus_file=effective_focus_file,
    )
    selected_file_paths = (
        file_result_set_selection.paths
        if file_result_set_selection is not None
        else None
    )
    selected_result_set_item_turn = bool(
        selected_file_paths is not None and len(selected_file_paths) == 1
    )
    result_set_comparison_turn = bool(
        selected_file_paths is not None
        and len(selected_file_paths) == 2
        and signals.result_set_comparison_followup
    )
    requires_result_set_generation = (
        selected_result_set_item_turn or result_set_comparison_turn
    )

    result_scope_paths: tuple[str, ...] | None = None
    if selected_file_paths is not None:
        result_scope_paths = selected_file_paths
    elif (
        effective_focus_file
        and state.last_result_set_entity_type == "文件"
        and event_name == "content_followup"
        and not signals.file_set_content_question
    ):
        result_scope_paths = (effective_focus_file,)

    if state.last_result_set_entity_type == "文件":
        if result_scope_paths is not None:
            query_result_set_items = result_scope_paths
            query_result_set_entity = "文件"
        elif (
            event_name
            in {
                "result_set_followup",
                "result_set_expansion_followup",
            }
            and state.last_result_set_items
        ):
            query_result_set_items = tuple(state.last_result_set_items)
            query_result_set_entity = "文件"
        elif (
            event_name
            in {
                "synthesis_request",
                "structured_request",
                "structured_skill_summary",
            }
            and state.last_result_set_focus_file
            and state.last_result_set_items
        ):
            query_result_set_items = tuple(state.last_result_set_items)
            query_result_set_entity = "文件"
        else:
            query_result_set_items = None
            query_result_set_entity = None
    else:
        query_result_set_items = (
            tuple(state.last_result_set_items)
            if state.last_result_set_items is not None
            else None
        )
        query_result_set_entity = state.last_result_set_entity_type

    has_single_focus_scope = bool(
        result_scope_paths
        and len(result_scope_paths) == 1
        and effective_focus_file
        and result_scope_paths[0] == effective_focus_file
        and event_name == "content_followup"
        and not signals.file_set_content_question
    )

    return ScopeDecision(
        clear_current_focus=signals.standalone_general_question,
        effective_focus_file=effective_focus_file,
        visible_file_paths=visible_file_paths,
        file_result_set_selection=file_result_set_selection,
        selected_file_paths=selected_file_paths,
        selected_result_set_item_turn=selected_result_set_item_turn,
        result_set_comparison_turn=result_set_comparison_turn,
        requires_result_set_generation=requires_result_set_generation,
        result_scope_paths=result_scope_paths,
        query_result_set_items=query_result_set_items,
        query_result_set_entity=query_result_set_entity,
        has_single_focus_scope=has_single_focus_scope,
    )
