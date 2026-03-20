from __future__ import annotations

import requests

from ai.prompt_builder import build_final_prompt
from ai.query_rewriter import rewrite_search_query
from retrieval.search_engine import (
    build_context_text,
    build_inventory_candidates_text,
    perform_retrieval,
)

from app.chat_text_utils import (
    build_clean_merged_query,
    build_timeline_evidence_text,
    extract_timeline_evidence_from_chunks,
    is_abstract_query,
    merge_rewritten_query_with_strong_terms,
    needs_timeline_evidence,
    redact_sensitive_text,
)


def _call_local_ollama(prompt: str, logger, ollama_api_url: str, ollama_model: str) -> str:
    logger.info("   🤖 正在使用本地模型做大方面概括...")
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(ollama_api_url, json=payload, timeout=180)
    response.raise_for_status()
    text = response.json().get("response", "").strip()
    logger.info(f"      ✨ 本地模型概括输出：[{text[:120]}]")
    return text


def build_topic_summarizer(logger, ollama_api_url: str, ollama_model: str):
    def _summarizer(prompt: str) -> str:
        return _call_local_ollama(
            prompt,
            logger=logger,
            ollama_api_url=ollama_api_url,
            ollama_model=ollama_model,
        )
    return _summarizer


def resolve_route(question: str, event, ollama_api_url: str, ollama_model: str, logger) -> str:
    if event.route_hint:
        route = event.route_hint
        logger.info(f"🧭 [状态机事件] {event.name}: {question} -> {route}")
        return route

    from ai.query_router import route_question

    route_info = route_question(question, ollama_api_url, ollama_model, logger)
    route = route_info["route"]
    logger.info(f"🧭 [模型路由] 问题: {question} -> {route}")
    return route


def build_search_query(
    *,
    question: str,
    event,
    flags: dict,
    memory_buffer: list[str],
    last_effective_search_query: str,
    logger,
    ollama_api_url: str,
    ollama_model: str,
) -> str:
    if flags["skip_retrieval"] or flags["is_inventory_query"]:
        return question

    base_query = question

    if event.merged_query:
        clean_merged_query = build_clean_merged_query(event.merged_query, question)
        logger.info(f"🧩 [状态机拼接检索] raw={event.merged_query}")
        logger.info(f"🧼 [模式词去壳后] merged={clean_merged_query}")
        base_query = clean_merged_query

    if is_abstract_query(question):
        prev_anchor = (last_effective_search_query or "").strip()
        if prev_anchor:
            base_query = f"{prev_anchor} {question}".strip()
            logger.info(f"🪝 [抽象追问继承] anchor={prev_anchor}")

    raw_search_query = rewrite_search_query(
        base_query,
        memory_buffer,
        ollama_api_url,
        ollama_model,
        logger,
    )

    search_query = merge_rewritten_query_with_strong_terms(question, raw_search_query)
    if not search_query.strip():
        search_query = (raw_search_query or base_query or question).strip()

    logger.info(f"🛡️ [强词保底后]：{search_query}")
    return search_query


def build_retrieval_materials(
    *,
    question: str,
    search_query: str,
    flags: dict,
    repo_state,
    model_emb,
    logger,
    current_focus_file,
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
        retrieval = perform_retrieval(
            question,
            search_query,
            repo_state,
            model_emb,
            logger,
            current_focus_file,
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
) -> str:
    safe_memory_buffer = [redact_sensitive_text(x) for x in memory_buffer]
    safe_inventory_candidates_text = redact_sensitive_text(inventory_candidates_text)
    safe_context_text = redact_sensitive_text(timeline_evidence_text + context_text)
    safe_question = redact_sensitive_text(question)

    return build_final_prompt(
        memory_buffer=safe_memory_buffer,
        current_focus_file=current_focus_file,
        inventory_candidates_text=safe_inventory_candidates_text,
        context_text=safe_context_text,
        question=safe_question,
    )