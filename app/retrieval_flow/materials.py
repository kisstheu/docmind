from __future__ import annotations

from ai.prompt_builder import build_final_prompt
from app.retrieval_flow.query import (
    filter_reused_indices_for_question,
    should_reuse_previous_results,
)
from app.chat_text_utils import (
    build_timeline_evidence_text,
    extract_timeline_evidence_from_chunks,
    needs_timeline_evidence,
    redact_sensitive_text,
)
from retrieval.search_engine import (
    build_context_text,
    build_inventory_candidates_text,
    perform_retrieval,
)


def build_retrieval_materials(
    *,
    question: str,
    search_query: str,
    context_anchor: str,
    flags: dict,
    repo_state,
    model_emb,
    logger,
    current_focus_file,
    last_relevant_indices=None,
    event=None,
):
    inventory_candidates_text = (
        build_inventory_candidates_text(question, repo_state, flags["inventory_target_type"])
        if flags["is_inventory_query"]
        else ""
    )

    context_text = ""
    timeline_evidence_text = ""
    relevant_indices = []

    if not flags["skip_retrieval"]:
        reuse_previous_results = should_reuse_previous_results(question, event, last_relevant_indices)

        if reuse_previous_results:
            logger.info("♻️ [追问复用] 使用上一轮检索结果，并按当前问题二次过滤")
            relevant_indices = filter_reused_indices_for_question(
                question=question,
                candidate_indices=last_relevant_indices,
                repo_state=repo_state,
                logger=logger,
            )
        else:
            retrieval = perform_retrieval(
                question,
                search_query,
                repo_state,
                model_emb,
                logger,
                current_focus_file,
                context_anchor=context_anchor,
            )
            current_focus_file = retrieval["current_focus_file"]
            relevant_indices = retrieval["relevant_indices"]

        context_text = build_context_text(relevant_indices, repo_state, logger)

        if needs_timeline_evidence(question):
            timeline_items = extract_timeline_evidence_from_chunks(
                relevant_indices,
                repo_state,
            )
            timeline_evidence_text = build_timeline_evidence_text(timeline_items)

    return {
        "inventory_candidates_text": inventory_candidates_text,
        "context_text": context_text,
        "timeline_evidence_text": timeline_evidence_text,
        "current_focus_file": current_focus_file,
        "relevant_indices": relevant_indices,
    }


def build_safe_final_prompt(
    *,
    memory_buffer: list[str],
    current_focus_file,
    inventory_candidates_text: str,
    context_text: str,
    timeline_evidence_text: str,
    question: str,
    event_name: str | None = None,
    result_set_items: list[str] | None = None,
) -> str:
    safe_memory_buffer = [redact_sensitive_text(x) for x in memory_buffer]
    safe_inventory_candidates_text = redact_sensitive_text(inventory_candidates_text)
    safe_context_text = redact_sensitive_text(timeline_evidence_text + context_text)
    safe_question = redact_sensitive_text(question)
    safe_result_set_items = [redact_sensitive_text(x) for x in (result_set_items or [])]

    constrained_context_text = safe_context_text
    constrained_question = safe_question

    if event_name == "result_set_followup" and safe_result_set_items:
        result_set_block = (
            "【上一轮候选集合】\n"
            + "\n".join(f"- {item}" for item in safe_result_set_items[:20])
            + "\n\n"
            "【结果集追问约束】\n"
            "当前问题是在上一轮候选集合基础上的进一步筛选。\n"
            "你只能在上述候选项中进行判断，不得新增集合外实体。\n"
            "若证据不足，可回答“无法确定”，不要扩展候选集合。\n\n"
        )
        constrained_context_text = result_set_block + constrained_context_text

    return build_final_prompt(
        memory_buffer=safe_memory_buffer,
        current_focus_file=current_focus_file,
        inventory_candidates_text=safe_inventory_candidates_text,
        context_text=constrained_context_text,
        question=safe_question,
        event_name=event_name,
        result_set_items=safe_result_set_items,
    )
