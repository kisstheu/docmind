from __future__ import annotations

import re
import warnings

from ai.decision_result import parse_decision_result
from ai.table_presentation import (
    StructuredTable,
    TableRenderOptions,
    bind_structured_table_row_identities,
)
from app.chat_state_answer_parsing import (
    EXPANSION_MARKERS,
    _contains_no_new_signal,
    _looks_like_file_locator_answer,
    _looks_like_source_backed_analytic_answer,
    extract_file_items,
    extract_numbered_items,
    infer_answer_type,
    infer_local_answer_type,
)
from app.chat_state_company_utils import (
    is_generic_company_reference,
    looks_like_real_company_name,
    normalize_company_item,
)
from app.dialog.question_scope import (
    QuestionSignals,
    ScopeDecision,
    analyze_question_signals,
    decide_file_result_set_scope,
)
from app.dialog.result_set import file_result_set_display_name, has_selectable_result_set

FOLLOWUP_EVENT_NAMES = {
    "entity_lookup_followup",
    "content_followup",
    "result_set_followup",
    "result_set_expansion_followup",
    "action_request",
    "judgment_request",
    "query_correction",
    "synthesis_request",
    "decision_request",
    "selected_candidate_followup",
}

_MISSING_QUESTION_SCOPE_FACT = object()


def _resolve_question_scope_facts(
    *,
    state,
    question: str,
    event_name: str | None,
    focused_file: str | None,
    question_signals: QuestionSignals | object,
    scope_decision: ScopeDecision | object,
) -> tuple[QuestionSignals, ScopeDecision]:
    missing_signals = question_signals is _MISSING_QUESTION_SCOPE_FACT
    missing_scope = scope_decision is _MISSING_QUESTION_SCOPE_FACT
    if missing_signals != missing_scope:
        raise TypeError(
            "question_signals and scope_decision must be provided together"
        )
    if missing_signals:
        warnings.warn(
            "Direct state-helper calls must pass QuestionSignals and ScopeDecision; "
            "the compatibility analysis path is deprecated.",
            DeprecationWarning,
            stacklevel=3,
        )
        resolved_signals = analyze_question_signals(
            question,
            last_effective_search_query=state.last_effective_search_query,
        )
        resolved_scope = decide_file_result_set_scope(
            question,
            signals=resolved_signals,
            state=state,
            current_focus_file=focused_file,
            event_name=(event_name or "").strip(),
        )
        return resolved_signals, resolved_scope
    if not isinstance(question_signals, QuestionSignals):
        raise TypeError("question_signals must be a QuestionSignals instance")
    if not isinstance(scope_decision, ScopeDecision):
        raise TypeError("scope_decision must be a ScopeDecision instance")
    return question_signals, scope_decision


def _extract_selected_candidate(question: str, answer_text: str) -> str | None:
    from app.dialog.task_semantics import is_recommendation_request

    if not is_recommendation_request(question):
        return None
    result = parse_decision_result(answer_text, user_question=question)
    return result.selected_candidate if result is not None else None

FOLLOWUP_HINT_MARKERS = [
    "吗",
    "呢",
    "呀",
    "继续",
    "还有",
    "再",
    "对应",
    "分别",
    "这",
    "那",
    "前面",
    "上面",
    "上述",
    "它们",
]


def append_memory(memory_buffer: list[str], question: str, answer: str, limit: int = 6) -> None:
    memory_buffer.append(f"用户问：{question}")
    memory_buffer.append(f"AI答：{answer}")
    if len(memory_buffer) > limit * 2:
        del memory_buffer[:-limit * 2]


def print_answer(answer_text: str, start_qa: float) -> None:
    import time

    print("\nAI回答：")
    print(answer_text)
    print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")


def update_state_after_answer_presentation(
    state,
    question: str,
    answer: str,
    *,
    table: StructuredTable | None = None,
    options: TableRenderOptions | None = None,
):
    """Replace the current view while preserving its factual authority."""
    if not getattr(state, "last_factual_answer_text", None):
        state.last_factual_answer_text = (
            state.last_answer_text or state.last_answer_preview or ""
        )
    state.last_user_question = question
    state.last_answer_text = answer
    state.last_answer_preview = answer[:200]
    state.last_answer_strategy = "last_answer_transform"
    if (
        table is not None
        and state.last_result_set_entity_type == "文件"
        and state.last_result_set_items
        and state.last_result_set_selectable is True
    ):
        display_name_counts: dict[str, int] = {}
        for item in state.last_result_set_items:
            display_name = file_result_set_display_name(item)
            display_name_counts[display_name] = display_name_counts.get(display_name, 0) + 1
        identity_aliases = {
            item: (
                (item, file_result_set_display_name(item))
                if display_name_counts[file_result_set_display_name(item)] == 1
                else (item,)
            )
            for item in state.last_result_set_items
        }
        table = bind_structured_table_row_identities(table, identity_aliases)
    state.current_presentation_table = table
    state.current_presentation_options = options if table is not None else None
    return state


def _merge_result_set_items(prev_items: list[str] | None, new_items: list[str] | None, limit: int = 20) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()

    for raw in (prev_items or []) + (new_items or []):
        item = (raw or "").strip()
        if not item:
            continue
        key = re.sub(r"\s+", "", item).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break

    return merged


def _is_followup_turn(question: str, event_name: str | None = None) -> bool:
    if (event_name or "").strip() in FOLLOWUP_EVENT_NAMES:
        return True

    q = (question or "").strip()
    if not q:
        return False

    if len(q) <= 16 and any(marker in q for marker in FOLLOWUP_HINT_MARKERS):
        return True

    if re.search(r"这[一二两三四五六七八九十\d]+", q):
        return True

    return False


def _looks_like_short_file_result_set_retry(question: str) -> bool:
    q = re.sub(r"[，。！？\s]+", "", (question or ""))
    if not q or len(q) > 16:
        return False
    return any(marker in q for marker in ("什么", "内容", "主题", "讲", "说"))


def _clear_generated_result_state(state) -> None:
    state.last_generated_result_items = None
    state.last_generated_result_entity_type = None
    state.last_generated_result_source_candidates = None
    state.last_generated_result_source_hits = None
    state.last_generated_result_focuses = None


def update_state_after_local_answer(
    state,
    question: str,
    answer: str,
    route: str,
    local_topic: str | None,
    is_content_answer: bool,
    *,
    canonical_file_paths: list[str] | tuple[str, ...] | None = None,
):
    state.last_user_question = question
    state.last_route = route
    state.last_local_topic = local_topic
    state.last_answer_preview = answer[:200]
    state.last_answer_text = answer
    state.last_factual_answer_text = answer
    state.current_presentation_table = None
    state.current_presentation_options = None
    state.last_answer_strategy = "local"
    state.last_answer_source_files = None
    if local_topic in {"category_summary", "category_count_breakdown", "category_overview"}:
        state.last_category_context_answer = answer

    if is_content_answer:
        state.last_content_user_question = question
        state.last_content_route = route
        state.last_content_topic = local_topic

    local_answer_type = infer_local_answer_type(question, answer, local_topic)
    state.last_answer_type = local_answer_type

    if local_answer_type == "enumeration_file" or local_topic in {"list_files", "list_files_by_topic"}:
        if canonical_file_paths is not None and local_topic in {"list_files", "list_files_by_topic"}:
            # Local inventories render complete paths. Bind each visible row to
            # the existing identity; filename parsing must not create authority.
            canonical_by_path = {path: path for path in canonical_file_paths}
            visible_items = extract_numbered_items(answer)
            selectable = bool(visible_items) and all(
                item in canonical_by_path for item in visible_items
            ) and len(set(visible_items)) == len(visible_items)
            file_items = (
                [canonical_by_path[item] for item in visible_items]
                if selectable else visible_items
            )
            # Unmapped rows retain display context only, with ordinals disabled.
            state.last_answer_type = "enumeration_file" if file_items else None
        else:
            file_items = extract_file_items(answer)
            selectable = bool(file_items)
        state.last_result_set_items = file_items or None
        state.last_result_set_entity_type = "文件" if file_items else None
        state.last_result_set_summary_text = None
        state.last_result_set_summary_level = 0
        state.last_result_set_selectable = selectable
        state.last_result_set_focus_file = None
        _clear_generated_result_state(state)

    return state


def update_state_after_local_file_result_set_answer(
    state,
    *,
    question: str,
    answer: str,
    items: list[str] | tuple[str, ...],
):
    """Write back a local file-set display operation while preserving its context."""
    previous_local_topic = state.last_local_topic
    previous_content_context = (
        state.last_content_route,
        state.last_content_user_question,
        state.last_content_topic,
    )
    previous_focus_file = state.last_result_set_focus_file
    previous_summary = (
        state.last_result_set_summary_text,
        state.last_result_set_summary_level,
    )

    state = update_state_after_local_answer(
        state,
        question=question,
        answer=answer,
        route=state.last_route or "normal_retrieval",
        local_topic=previous_local_topic,
        is_content_answer=False,
    )
    state.last_local_topic = previous_local_topic
    (
        state.last_content_route,
        state.last_content_user_question,
        state.last_content_topic,
    ) = previous_content_context
    state.last_result_set_items = list(items)
    state.last_result_set_entity_type = "文件"
    state.last_result_set_selectable = True
    state.last_result_set_focus_file = previous_focus_file
    (
        state.last_result_set_summary_text,
        state.last_result_set_summary_level,
    ) = previous_summary
    state.last_answer_type = "enumeration_file"
    return state


def update_state_after_retrieval_answer(
    state,
    question: str,
    answer_text: str,
    logger,
    event_name: str | None = None,
    focused_file: str | None = None,
    decision_result=None,
    *,
    question_signals: QuestionSignals | object = _MISSING_QUESTION_SCOPE_FACT,
    scope_decision: ScopeDecision | object = _MISSING_QUESTION_SCOPE_FACT,
    generated_result_provenance=None,
    answer_strategy: str = "content",
):
    question_signals, scope_decision = _resolve_question_scope_facts(
        state=state,
        question=question,
        event_name=event_name,
        focused_file=focused_file,
        question_signals=question_signals,
        scope_decision=scope_decision,
    )
    prev_result_set_items = list(state.last_result_set_items) if state.last_result_set_items else None
    prev_result_set_entity_type = state.last_result_set_entity_type
    prev_answer_type = state.last_answer_type
    prev_result_set_summary_text = state.last_result_set_summary_text
    prev_result_set_summary_level = state.last_result_set_summary_level
    prev_result_set_selectable = state.last_result_set_selectable
    prev_result_set_focus_file = state.last_result_set_focus_file
    is_followup_turn = _is_followup_turn(question, event_name=event_name)
    is_synthesis_answer = (event_name or "").strip() == "synthesis_request"
    is_focus_related_event = (
        (event_name or "").strip() in FOLLOWUP_EVENT_NAMES
        or scope_decision.selected_result_set_item_turn
    )
    clears_result_set_focus = (
        question_signals.standalone_general_question
        and not scope_decision.selected_result_set_item_turn
    )
    preserves_active_file_result_set = (
        prev_result_set_entity_type == "文件"
        and bool(prev_result_set_items)
        and not clears_result_set_focus
    )

    state.last_user_question = question
    state.last_route = "normal_retrieval"
    state.last_local_topic = None

    state.last_content_user_question = question
    state.last_content_route = "normal_retrieval"
    state.last_content_topic = None

    state.last_answer_text = answer_text
    state.last_answer_preview = answer_text[:200]
    state.last_factual_answer_text = answer_text
    state.current_presentation_table = None
    state.current_presentation_options = None
    state.last_answer_strategy = answer_strategy
    state.last_answer_source_files = (
        extract_file_items(answer_text)
        if answer_strategy == "direct_evidence"
        else None
    )

    structured_decision = decision_result
    if structured_decision is None and (event_name or "").strip() == "decision_request":
        structured_decision = parse_decision_result(answer_text, user_question=question)
    selected_candidate = getattr(structured_decision, "selected_candidate", None)
    is_comparison_decision = bool(getattr(structured_decision, "comparison_requested", False))
    if is_comparison_decision:
        state.last_answer_source_files = list(structured_decision.source_files) or None
    if selected_candidate:
        selected_sources = list(
            structured_decision.selected_source_files
            if is_comparison_decision
            else getattr(structured_decision, "source_files", ()) or ()
        )
        state.last_selected_candidate = selected_candidate
        state.last_selected_source_files = selected_sources or None
        logger.debug(
            f"🧪 [选择状态] candidate={selected_candidate} | sources={state.last_selected_source_files}"
        )
    elif is_comparison_decision or (event_name or "").strip() == "decision_request":
        state.last_selected_candidate = None
        state.last_selected_source_files = None
        logger.debug("🧪 [选择状态] 本轮未形成可靠选择，不写入推荐焦点")

    inferred_answer_type = infer_answer_type(question, answer_text)
    is_result_set_comparison_answer = (
        prev_result_set_entity_type == "文件"
        and bool(prev_result_set_items)
        and question_signals.result_set_comparison_followup
    )
    is_result_set_item_answer = (
        prev_result_set_entity_type == "文件"
        and bool(prev_result_set_items)
        and bool(focused_file)
        and not is_result_set_comparison_answer
        and (
            question_signals.explicit_single_file_result_reference
            or scope_decision.has_single_focus_scope
        )
    )
    answer_type = (
        None
        if is_synthesis_answer or is_comparison_decision or is_result_set_comparison_answer or is_result_set_item_answer
        else inferred_answer_type
    )
    state.last_answer_type = answer_type
    if is_synthesis_answer and inferred_answer_type is not None:
        logger.debug(
            f"🧪 [answer_type识别] 综合回答忽略文本外观类型={inferred_answer_type}，保持内容焦点"
        )
    elif is_result_set_item_answer and inferred_answer_type is not None:
        logger.debug(
            f"🧪 [answer_type识别] 结果集单项内容回答忽略文本外观类型={inferred_answer_type}"
        )

    current_result_set_focus_file = focused_file or prev_result_set_focus_file
    structured_enumeration_attempted = bool(
        getattr(generated_result_provenance, "enumeration_attempted", False)
    )
    structured_provenance_items = list(
        getattr(generated_result_provenance, "display_items", ()) or ()
    )
    reliable_structured_enumeration = bool(
        structured_enumeration_attempted
        and getattr(generated_result_provenance, "reliable", False)
        and extract_numbered_items(answer_text) == structured_provenance_items
    )
    prev_generated_focuses = list(state.last_generated_result_focuses or [])
    has_prev_materialized_generated_result_set = bool(
        prev_result_set_items
        and state.last_generated_result_items == prev_result_set_items
        and len(prev_generated_focuses) == len(prev_result_set_items)
        and all(prev_generated_focuses)
    )

    if structured_enumeration_attempted:
        display_items = structured_provenance_items
        source_candidates = list(
            getattr(generated_result_provenance, "source_candidates", ()) or ()
        )
        source_hits = [
            list(paths)
            for paths in (
                getattr(generated_result_provenance, "source_hits", ()) or ()
            )
        ]
        opaque_focuses = list(
            getattr(generated_result_provenance, "opaque_focuses", ()) or ()
        )
        entity_type = str(
            getattr(generated_result_provenance, "entity_type", None) or ""
        ).strip()
        if reliable_structured_enumeration:
            state.last_answer_type = "enumeration_generated"
            state.last_result_set_query = question
            state.last_result_set_items = display_items
            state.last_result_set_entity_type = entity_type
            state.last_result_set_selectable = True
            state.last_result_set_focus_file = None
            state.last_result_set_summary_text = None
            state.last_result_set_summary_level = 0
            state.last_generated_result_items = display_items
            state.last_generated_result_entity_type = entity_type
            state.last_generated_result_source_candidates = source_candidates
            state.last_generated_result_source_hits = source_hits
            state.last_generated_result_focuses = opaque_focuses
            state.last_selected_candidate = None
            state.last_selected_source_files = None
            logger.debug(
                "🧭 [生成结果集物化] "
                f"entity={entity_type} | items={display_items} | selectable=True"
            )
        else:
            state.last_answer_type = None
            state.last_result_set_items = prev_result_set_items
            state.last_result_set_entity_type = prev_result_set_entity_type
            state.last_result_set_selectable = False
            state.last_result_set_focus_file = prev_result_set_focus_file
            state.last_generated_result_items = display_items or None
            state.last_generated_result_entity_type = entity_type if display_items else None
            state.last_generated_result_source_candidates = source_candidates or (
                list(prev_result_set_items or []) or None
            )
            state.last_generated_result_source_hits = (
                [[] for _item in display_items]
            ) or None
            state.last_generated_result_focuses = None
            logger.debug(
                "🛡️ [结果集序号安全] 结构化枚举未通过全量绑定校验，"
                "保留旧结果集上下文但撤销序号选择资格"
            )
            logger.debug(
                "🧭 [生成结果来源] "
                f"display_items={state.last_generated_result_items} | "
                f"source_hits={state.last_generated_result_source_hits} | "
                f"binding_failures={list(getattr(generated_result_provenance, 'binding_failures', ()) or ())}"
            )
    elif answer_type == "enumeration_company":
        _clear_generated_result_state(state)
        company_items: list[str] = []
        raw_items = extract_numbered_items(answer_text)

        for item in raw_items:
            cleaned = normalize_company_item(item)
            if not cleaned:
                continue
            if is_generic_company_reference(cleaned):
                continue
            if looks_like_real_company_name(cleaned):
                company_items.append(cleaned)
                continue
            company_items.append(cleaned)

        if not company_items and raw_items:
            for item in raw_items:
                cleaned = normalize_company_item(item)
                if len(cleaned) >= 2 and cleaned not in company_items:
                    company_items.append(cleaned)

        deduped_items: list[str] = []
        seen_norm: set[str] = set()
        for item in company_items:
            norm_key = re.sub(r"\s+", "", item).lower()
            if not norm_key or norm_key in seen_norm:
                continue
            seen_norm.add(norm_key)
            deduped_items.append(item)
        company_items = deduped_items

        if prev_result_set_entity_type == "公司" and prev_result_set_items:
            if company_items and is_followup_turn:
                company_items = _merge_result_set_items(prev_result_set_items, company_items)
                logger.debug("🧪 [候选集合提取] 追问场景合并公司候选集合")
            elif not company_items and _contains_no_new_signal(answer_text):
                company_items = prev_result_set_items
                logger.debug("🧪 [候选集合提取] 本轮无新增公司，沿用上一轮公司候选集合")

        state.last_result_set_items = company_items
        state.last_result_set_entity_type = "公司"
        state.last_result_set_selectable = has_selectable_result_set(company_items, "公司", answer_text)
        if not state.last_result_set_selectable and company_items == prev_result_set_items:
            state.last_result_set_selectable = prev_result_set_selectable

        logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={answer_type}")
        logger.debug(f"🧪 [候选集合提取] raw_items={raw_items}")
        logger.debug(f"🧪 [候选集合提取] company_items={company_items}")
    elif answer_type == "enumeration_file":
        _clear_generated_result_state(state)
        file_items = extract_file_items(answer_text)

        if prev_result_set_entity_type == "文件" and prev_result_set_items:
            if file_items and is_followup_turn:
                file_items = _merge_result_set_items(prev_result_set_items, file_items)
                logger.debug("🧪 [候选集合提取] 追问场景合并文件候选集合")
            elif not file_items and preserves_active_file_result_set:
                file_items = prev_result_set_items
                logger.debug(
                    "🧪 [候选集合提取] 本轮未形成有效新文件集合且无明确失效依据，"
                    "沿用上一轮文件候选集合"
                )

        state.last_result_set_items = file_items or None
        state.last_result_set_entity_type = "文件" if file_items else None
        state.last_result_set_summary_text = None
        state.last_result_set_summary_level = 0
        state.last_result_set_selectable = has_selectable_result_set(file_items, "文件", answer_text)
        state.last_result_set_focus_file = None
        if not state.last_result_set_selectable and file_items == prev_result_set_items:
            state.last_result_set_selectable = prev_result_set_selectable

        logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={answer_type}")
        logger.debug(f"🧪 [候选集合提取] file_items={file_items}")
    elif answer_type == "enumeration_person":
        _clear_generated_result_state(state)
        person_items = extract_numbered_items(answer_text)

        if prev_result_set_entity_type == "人物" and prev_result_set_items:
            if person_items and is_followup_turn:
                person_items = _merge_result_set_items(prev_result_set_items, person_items)
                logger.debug("🧪 [候选集合提取] 追问场景合并人物候选集合")
            elif not person_items and _contains_no_new_signal(answer_text):
                person_items = prev_result_set_items
                logger.debug("🧪 [候选集合提取] 本轮无新增人物，沿用上一轮人物候选集合")

        state.last_result_set_items = person_items
        state.last_result_set_entity_type = "人物"
        state.last_result_set_selectable = has_selectable_result_set(person_items, "人物", answer_text)
        state.last_result_set_focus_file = None
        if not state.last_result_set_selectable and person_items == prev_result_set_items:
            state.last_result_set_selectable = prev_result_set_selectable

        logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={answer_type}")
        logger.debug(f"🧪 [候选集合提取] person_items={person_items}")

    else:
        q_norm = re.sub(r"[，。！？\s]+", "", (question or ""))
        entity_to_answer_type = {
            "公司": "enumeration_company",
            "文件": "enumeration_file",
            "人物": "enumeration_person",
        }
        fallback_file_items = extract_file_items(answer_text)
        full_file_result_set_content_scope = (
            prev_result_set_entity_type == "文件"
            and bool(prev_result_set_items)
            and tuple(scope_decision.result_scope_paths or ())
            == tuple(prev_result_set_items or ())
        )
        preserve_source_file_refs = (
            not is_synthesis_answer
            and not preserves_active_file_result_set
            and bool(fallback_file_items)
            and _looks_like_source_backed_analytic_answer(question, answer_text)
        )
        fallback_to_file_result_set = (
            not is_synthesis_answer
            and not scope_decision.has_single_focus_scope
            and not full_file_result_set_content_scope
            and bool(fallback_file_items)
            and not preserve_source_file_refs
            and _looks_like_file_locator_answer(answer_text)
            and (is_followup_turn or "文件" in question or "文档" in question or "记录" in question)
        )

        keep_result_set_context = (
            prev_result_set_entity_type in entity_to_answer_type
            and bool(prev_result_set_items)
            and any(x in q_norm for x in EXPANSION_MARKERS)
            and _contains_no_new_signal(answer_text or "")
        )
        preserve_result_set_on_result_set_followup = (
            (event_name or "").strip() in {"result_set_followup", "result_set_expansion_followup"}
            and (
                prev_result_set_entity_type in entity_to_answer_type
                or has_prev_materialized_generated_result_set
            )
            and bool(prev_result_set_items)
        )
        file_result_set_content_operation = (
            preserve_result_set_on_result_set_followup
            and full_file_result_set_content_scope
        )
        preserve_parent_result_set_scope = (
            (
                prev_result_set_entity_type in entity_to_answer_type
                or has_prev_materialized_generated_result_set
            )
            and bool(prev_result_set_items)
            and scope_decision.result_scope_paths is not None
            and is_followup_turn
        )
        preserve_materialized_generated_selection = bool(
            has_prev_materialized_generated_result_set
            and state.last_selected_candidate
            and (
                (event_name or "").strip() in FOLLOWUP_EVENT_NAMES
                or scope_decision.selected_result_set_item_turn
            )
        )
        generated_unmaterialized_enumeration = (
            preserve_result_set_on_result_set_followup
            and not scope_decision.requires_result_set_generation
            and not file_result_set_content_operation
            and bool(extract_numbered_items(answer_text))
        )
        preserve_file_scope_on_content_question = (
            file_result_set_content_operation
        )
        preserve_file_scope_on_synthesis = (
            is_synthesis_answer
            and prev_result_set_entity_type == "文件"
            and bool(prev_result_set_items)
        )
        preserve_file_scope_on_detail_followup = (
            preserve_result_set_on_result_set_followup
            and prev_result_set_entity_type == "文件"
            and question_signals.detail_explanation_request
        )
        preserve_file_focus_context = (
            prev_result_set_entity_type == "文件"
            and bool(prev_result_set_items)
            and bool(current_result_set_focus_file)
            and is_focus_related_event
            and not clears_result_set_focus
            and not question_signals.all_items_file_set_content_question
            and (
                scope_decision.has_single_focus_scope
                or (event_name or "").strip() == "content_followup"
                or question_signals.content_followup_question
                or question_signals.explicit_focus_reference
                or question_signals.explicit_single_file_result_reference
                or question_signals.detail_explanation_request
                or question_signals.context_dependent_question
            )
        )
        preserve_file_result_set_on_summary_followup = (
            prev_result_set_entity_type == "文件"
            and bool(prev_result_set_items)
            and question_signals.summary_followup_request
        )
        preserve_file_result_set_on_no_evidence_followup = (
            prev_result_set_entity_type == "文件"
            and bool(prev_result_set_items)
            and (is_followup_turn or _looks_like_short_file_result_set_retry(question))
            and _contains_no_new_signal(answer_text or "")
        )

        if fallback_to_file_result_set:
            if prev_result_set_entity_type == "文件" and prev_result_set_items and is_followup_turn:
                fallback_file_items = _merge_result_set_items(prev_result_set_items, fallback_file_items)
                logger.debug("🧪 [状态保留] 文件定位回答触发追问合并，保留并扩充上一轮文件候选集合")

            state.last_result_set_items = fallback_file_items
            state.last_result_set_entity_type = "文件"
            state.last_answer_type = "enumeration_file"
            state.last_result_set_summary_text = None
            state.last_result_set_summary_level = 0
            state.last_result_set_selectable = False
            state.last_result_set_focus_file = None
            logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={state.last_answer_type}")
            logger.debug(f"🧪 [候选集合提取] file_items={fallback_file_items}")
        elif preserve_source_file_refs:
            if prev_result_set_entity_type is None and prev_result_set_items and is_followup_turn:
                fallback_file_items = _merge_result_set_items(prev_result_set_items, fallback_file_items)
                logger.debug("🧪 [状态保留] 分析回答附带来源文件，合并上一轮来源文件候选集合")

            state.last_result_set_items = fallback_file_items
            state.last_result_set_entity_type = None
            state.last_answer_type = None
            state.last_result_set_selectable = False
            state.last_result_set_focus_file = None
            logger.debug("🧪 [状态保留] 分析回答仅保留来源文件候选，不视为文件结果集")
            logger.debug(f"🧪 [候选集合提取] analytic_source_file_items={fallback_file_items}")
        elif preserve_file_focus_context:
            state.last_result_set_items = prev_result_set_items
            state.last_result_set_entity_type = prev_result_set_entity_type
            state.last_result_set_selectable = prev_result_set_selectable
            state.last_result_set_focus_file = current_result_set_focus_file
            state.last_answer_type = None
            logger.debug(
                f"🧪 [状态保留] 当前文件焦点延续 focus={state.last_result_set_focus_file}"
            )
        elif (
            keep_result_set_context
            or preserve_result_set_on_result_set_followup
            or preserve_file_scope_on_synthesis
            or preserve_file_scope_on_detail_followup
            or preserve_file_result_set_on_summary_followup
            or preserve_file_result_set_on_no_evidence_followup
            or preserve_parent_result_set_scope
            or preserve_materialized_generated_selection
            or preserves_active_file_result_set
        ):
            state.last_result_set_items = prev_result_set_items
            state.last_result_set_entity_type = prev_result_set_entity_type
            state.last_result_set_focus_file = (
                prev_result_set_focus_file
                if has_prev_materialized_generated_result_set
                else current_result_set_focus_file or prev_result_set_focus_file
            )
            if preserve_file_scope_on_detail_followup:
                state.last_answer_type = None
                logger.debug("🧪 [状态保留] 文件结果集展开回答保留原范围，不写成文件枚举")
            elif preserve_file_scope_on_content_question or preserve_file_scope_on_synthesis:
                state.last_answer_type = None
                if preserve_file_scope_on_synthesis:
                    state.last_result_set_selectable = False
                state.last_result_set_summary_text = answer_text.strip()
                state.last_result_set_summary_level = max(1, prev_result_set_summary_level + 1)
                logger.debug("🧪 [状态保留] 文件集合综合回答保留原范围，但不写成文件枚举")
            elif preserve_file_result_set_on_summary_followup:
                state.last_answer_type = answer_type
                state.last_result_set_summary_text = answer_text.strip()
                state.last_result_set_summary_level = (
                    max(1, prev_result_set_summary_level + 1)
                    if prev_result_set_summary_text
                    else 1
                )
                logger.debug("🧪 [状态保留] 文件结果集概括未产出新集合，保留候选文件但清除枚举回答类型")
            elif preserve_materialized_generated_selection:
                state.last_answer_type = None
                state.last_result_set_selectable = True
                logger.debug(
                    "🧪 [状态保留] 生成结果集单项回答保留对象集合与选择焦点"
                )
            elif preserve_parent_result_set_scope:
                state.last_answer_type = None
                state.last_result_set_selectable = prev_result_set_selectable
                logger.debug("🧪 [状态保留] 受限内容追问未产出新集合，保留父结果集范围")
            elif preserve_file_result_set_on_no_evidence_followup:
                state.last_answer_type = prev_answer_type
                state.last_result_set_selectable = prev_result_set_selectable
                logger.debug("🧪 [状态保留] 文件短追问未拿到新证据，继续沿用上一轮文件结果集")
            elif preserves_active_file_result_set:
                state.last_answer_type = None
                state.last_result_set_selectable = prev_result_set_selectable
                logger.debug("🧪 [状态保留] 普通内容回答没有明确失效依据，保留活动文件结果集")
            else:
                state.last_answer_type = prev_answer_type or entity_to_answer_type.get(prev_result_set_entity_type)
            if not preserve_file_result_set_on_summary_followup and not preserve_file_scope_on_content_question and not preserve_file_scope_on_synthesis:
                if preserve_result_set_on_result_set_followup:
                    logger.debug(
                        f"🧪 [状态保留] 结果集追问回答未产出新集合，保留 entity={prev_result_set_entity_type}"
                    )
                elif preserve_file_result_set_on_no_evidence_followup:
                    logger.debug("🧪 [状态保留] 文件短追问未拿到新证据，继续保留上一轮文件结果集")
                else:
                    logger.debug(
                        f"🧪 [状态保留] 扩展追问未新增，保留 entity={prev_result_set_entity_type}"
                    )
            logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={state.last_answer_type}")
        else:
            state.last_result_set_items = None
            state.last_result_set_entity_type = None
            state.last_result_set_selectable = False
            state.last_result_set_focus_file = None
            _clear_generated_result_state(state)
            logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={answer_type}")

        if generated_unmaterialized_enumeration and state.last_result_set_items:
            state.last_answer_type = None
            state.last_result_set_selectable = False
            visible_items = extract_numbered_items(answer_text)
            provenance_items = list(
                getattr(generated_result_provenance, "display_items", ()) or ()
            )
            provenance_candidates = list(
                getattr(generated_result_provenance, "source_candidates", ()) or ()
            )
            provenance_hits = [
                list(paths)
                for paths in (
                    getattr(generated_result_provenance, "source_hits", ()) or ()
                )
            ]
            state.last_generated_result_items = list(visible_items)
            state.last_generated_result_entity_type = None
            state.last_generated_result_source_candidates = (
                provenance_candidates or list(prev_result_set_items or [])
            )
            state.last_generated_result_source_hits = (
                provenance_hits
                if provenance_items == visible_items
                and len(provenance_hits) == len(visible_items)
                else [[] for _item in visible_items]
            )
            logger.debug(
                "🛡️ [结果集序号安全] 当前可见枚举未物化可靠内部集合，"
                "保留旧结果集上下文但撤销序号选择资格"
            )
            logger.debug(
                "🧭 [生成结果来源] "
                f"display_items={state.last_generated_result_items} | "
                f"source_hits={state.last_generated_result_source_hits}"
            )

    if state.last_result_set_entity_type != "文件":
        state.last_result_set_summary_text = None
        state.last_result_set_summary_level = 0

    should_write_result_set_focus = (
        bool(focused_file)
        and not clears_result_set_focus
        and not has_prev_materialized_generated_result_set
        and (
            scope_decision.has_single_focus_scope
            or (event_name or "").strip() == "content_followup"
            or question_signals.explicit_focus_reference
            or question_signals.explicit_single_file_result_reference
        )
    )
    if should_write_result_set_focus:
        if (
            state.last_result_set_entity_type == "文件"
            and state.last_result_set_items
            and (prev_result_set_focus_file or is_focus_related_event)
        ):
            state.last_result_set_focus_file = focused_file
            logger.debug(f"🧪 [文件焦点状态] focus={focused_file} | 保留结果集并延续单文件内容语义")
        else:
            state.last_answer_type = None
            state.last_result_set_items = None
            state.last_result_set_entity_type = None
            state.last_result_set_selectable = False
            state.last_result_set_focus_file = None
            state.last_result_set_summary_text = None
            state.last_result_set_summary_level = 0
            logger.debug(f"🧪 [文件焦点状态] focus={focused_file} | 清除旧结果集，保留单文件内容语义")

    logger.debug(
        f"🧠 [状态写回] "
        f"last_user_question={state.last_user_question} | "
        f"last_content_route={state.last_content_route} | "
        f"last_answer_type={state.last_answer_type}"
        f" | result_set_entity={state.last_result_set_entity_type}"
        f" | result_set_items={state.last_result_set_items}"
    )

    return state
