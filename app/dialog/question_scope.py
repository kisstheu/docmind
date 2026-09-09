from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from app.context_anchor import is_context_dependent_question
from app.dialog.result_set import (
    FileResultSetSelection,
    has_explicit_single_file_result_reference,
    has_selectable_result_set,
    looks_like_result_set_comparison_followup,
    looks_like_result_set_predicate_followup,
    reject_unmapped_file_result_set_ordinal,
    resolve_generated_result_set_selection,
    resolve_file_result_set_selection,
    resolve_presentation_file_result_set_selection,
)
from app.dialog.repo_meta_rules import (
    has_explicit_content_enumeration_subject,
    is_collection_context_open_enumeration_request,
    is_explicit_corpus_content_enumeration_request,
)
from app.dialog.task_semantics import (
    is_collection_synthesis_request,
    is_detail_explanation_request,
    is_document_evaluation_request,
    is_subjectless_collection_synthesis_request,
)
from app.dialog_utils import is_content_followup_question, is_summary_followup_request
from app.chat_text.core import is_answer_depth_followup
from app.chat_text.file_lookup import (
    has_explicit_focus_reference,
    looks_like_bare_content_question,
    looks_like_all_items_file_set_content_question,
    looks_like_file_set_content_question,
    looks_like_implicit_file_set_content_question,
    looks_like_standalone_general_question,
)


class _QuestionScopeState(Protocol):
    last_result_set_items: list[str] | None
    last_result_set_entity_type: str | None
    last_answer_text: str | None
    last_answer_preview: str | None
    last_result_set_selectable: bool | None
    last_result_set_focus_file: str | None
    last_result_set_summary_text: str | None
    last_generated_result_items: list[str] | None
    last_generated_result_entity_type: str | None
    last_generated_result_source_candidates: list[str] | None
    last_generated_result_source_hits: list[list[str]] | None
    last_generated_result_focuses: list[str] | None
    last_selected_candidate: str | None
    current_presentation_table: object | None


@dataclass(frozen=True)
class QuestionSignals:
    standalone_general_question: bool
    file_set_content_question: bool
    implicit_file_set_content_question: bool
    bare_content_question: bool
    all_items_file_set_content_question: bool
    explicit_focus_reference: bool
    explicit_single_file_result_reference: bool
    result_set_comparison_followup: bool
    content_followup_question: bool
    detail_explanation_request: bool
    answer_depth_followup: bool
    summary_followup_request: bool
    context_dependent_question: bool
    document_evaluation_request: bool
    result_set_predicate_followup: bool


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
    answer_entity_followup: bool


def analyze_question_signals(
    question: str,
    *,
    last_effective_search_query: str | None,
) -> QuestionSignals:
    explicit_focus_reference = has_explicit_focus_reference(question)
    explicit_single_file_reference = has_explicit_single_file_result_reference(
        question
    )
    standalone_general_question = bool(
        looks_like_standalone_general_question(question)
        or (
            has_explicit_content_enumeration_subject(question)
            and not explicit_focus_reference
            and not explicit_single_file_reference
        )
    )
    return QuestionSignals(
        standalone_general_question=standalone_general_question,
        file_set_content_question=looks_like_file_set_content_question(question),
        implicit_file_set_content_question=looks_like_implicit_file_set_content_question(
            question
        ),
        bare_content_question=looks_like_bare_content_question(question),
        all_items_file_set_content_question=looks_like_all_items_file_set_content_question(
            question
        ),
        explicit_focus_reference=explicit_focus_reference,
        explicit_single_file_result_reference=explicit_single_file_reference,
        result_set_comparison_followup=looks_like_result_set_comparison_followup(
            question
        ),
        content_followup_question=is_content_followup_question(question),
        detail_explanation_request=is_detail_explanation_request(question),
        answer_depth_followup=is_answer_depth_followup(question),
        summary_followup_request=is_summary_followup_request(question),
        context_dependent_question=is_context_dependent_question(
            question,
            last_effective_search_query,
        ),
        document_evaluation_request=is_document_evaluation_request(question),
        result_set_predicate_followup=looks_like_result_set_predicate_followup(
            question
        ),
    )


def decide_file_result_set_scope(
    question: str,
    *,
    signals: QuestionSignals,
    state: _QuestionScopeState,
    current_focus_file: str | None,
    event_name: str,
    corpus_paths: Iterable[str] | None = None,
) -> ScopeDecision:
    indexed_corpus_paths = tuple(
        dict.fromkeys(
            str(path or "").strip()
            for path in corpus_paths or ()
            if str(path or "").strip()
        )
    )
    active_collection_paths: tuple[str, ...] = ()
    if state.last_result_set_entity_type == "文件" and state.last_result_set_items:
        active_collection_paths = tuple(state.last_result_set_items)
    elif (
        state.last_generated_result_items
        and state.last_generated_result_source_candidates
        and not state.last_selected_candidate
    ):
        active_collection_paths = tuple(
            dict.fromkeys(state.last_generated_result_source_candidates)
        )
    collection_open_enumeration = is_collection_context_open_enumeration_request(
        question,
        has_collection_context=bool(active_collection_paths),
    )
    explicit_corpus_synthesis_scope = bool(
        event_name == "synthesis_request"
        and is_explicit_corpus_content_enumeration_request(question)
    )
    answer_entity_followup = bool(
        event_name in {"content_followup", "structured_request"}
        and state.last_generated_result_items
        and state.last_generated_result_entity_type
        and state.last_generated_result_source_candidates
        and not state.last_selected_candidate
        and not signals.standalone_general_question
        and not signals.explicit_single_file_result_reference
        and not signals.result_set_comparison_followup
    )
    collection_scope_continuation = bool(
        signals.answer_depth_followup
        and state.last_result_set_entity_type == "文件"
        and state.last_result_set_items
        and state.last_result_set_summary_text
    )
    collection_synthesis_scope = bool(
        event_name == "synthesis_request"
        and active_collection_paths
        and not explicit_corpus_synthesis_scope
        and (
            is_subjectless_collection_synthesis_request(question)
            or is_collection_synthesis_request(
                question,
                has_collection_context=False,
            )
            or collection_open_enumeration
        )
    )
    effective_focus_file = (
        None
        if (
            signals.standalone_general_question
            or explicit_corpus_synthesis_scope
            or collection_scope_continuation
            or collection_synthesis_scope
            or answer_entity_followup
        )
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

    current_presentation = getattr(state, "current_presentation_table", None)
    presentation_file_ordinal = bool(
        current_presentation is not None
        and state.last_result_set_entity_type == "文件"
        and state.last_result_set_items
        and signals.explicit_single_file_result_reference
    )
    if presentation_file_ordinal:
        file_result_set_selection = resolve_presentation_file_result_set_selection(
            question,
            getattr(current_presentation, "row_identities", None),
            visible_file_paths,
            focus_file=effective_focus_file,
        )
    else:
        file_result_set_selection = resolve_generated_result_set_selection(
            question,
            state.last_generated_result_items,
            state.last_generated_result_source_hits,
            state.last_generated_result_source_candidates,
            getattr(state, "last_generated_result_focuses", None),
        )
    if file_result_set_selection is None and (
        state.last_result_set_entity_type == "文件"
        and state.last_result_set_items
        and not visible_file_paths
    ):
        file_result_set_selection = reject_unmapped_file_result_set_ordinal(
            question
        )
    if file_result_set_selection is None:
        file_result_set_selection = resolve_file_result_set_selection(
            question,
            visible_file_paths,
            focus_file=effective_focus_file,
        )
    selected_file_paths = (
        file_result_set_selection.paths
        if (
            file_result_set_selection is not None
            and file_result_set_selection.rejection is None
        )
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
    elif explicit_corpus_synthesis_scope:
        result_scope_paths = indexed_corpus_paths
    elif (
        signals.all_items_file_set_content_question
        and state.last_result_set_entity_type == "文件"
        and state.last_result_set_items
    ):
        result_scope_paths = tuple(state.last_result_set_items)
    elif (
        signals.bare_content_question
        and not effective_focus_file
        and state.last_result_set_entity_type == "文件"
        and state.last_result_set_items
    ):
        result_scope_paths = tuple(state.last_result_set_items)
    elif collection_scope_continuation:
        result_scope_paths = tuple(state.last_result_set_items or ())
    elif collection_synthesis_scope:
        result_scope_paths = active_collection_paths
    elif (
        event_name == "result_set_followup"
        and active_collection_paths
        and state.last_result_set_entity_type == "文件"
        and signals.result_set_predicate_followup
    ):
        result_scope_paths = active_collection_paths
    elif answer_entity_followup:
        result_scope_paths = tuple(state.last_generated_result_source_candidates or ())
    elif (
        effective_focus_file
        and event_name == "content_followup"
        and not signals.all_items_file_set_content_question
    ):
        result_scope_paths = (effective_focus_file,)

    if explicit_corpus_synthesis_scope:
        query_result_set_items = indexed_corpus_paths
        query_result_set_entity = "文件"
    elif collection_synthesis_scope:
        query_result_set_items = active_collection_paths
        query_result_set_entity = "文件"
    elif answer_entity_followup:
        query_result_set_items = tuple(state.last_generated_result_items or ())
        query_result_set_entity = state.last_generated_result_entity_type
    elif state.last_result_set_entity_type == "文件":
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
    elif (
        file_result_set_selection is not None
        and file_result_set_selection.display_item
        and file_result_set_selection.opaque_focus
    ):
        query_result_set_items = (file_result_set_selection.display_item,)
        query_result_set_entity = state.last_result_set_entity_type
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
        and not signals.all_items_file_set_content_question
    )
    whole_file_result_set_scope = bool(
        result_scope_paths
        and len(result_scope_paths) > 1
        and result_scope_paths == active_collection_paths
        and event_name == "result_set_followup"
    )

    return ScopeDecision(
        clear_current_focus=(
            signals.standalone_general_question
            or explicit_corpus_synthesis_scope
            or collection_scope_continuation
            or collection_synthesis_scope
            or answer_entity_followup
            or whole_file_result_set_scope
        ),
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
        answer_entity_followup=answer_entity_followup,
    )
