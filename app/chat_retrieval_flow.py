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
    build_timeline_evidence_text,
    extract_strong_terms_from_question,
    extract_timeline_evidence_from_chunks,
    is_result_expansion_followup,
    keep_only_current_question_terms,
    merge_rewritten_query_with_strong_terms,
    needs_timeline_evidence,
    normalize_question_for_retrieval,
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


def should_reuse_previous_results(question: str, event, last_relevant_indices) -> bool:
    return (
        getattr(event, "name", "") == "content_followup"
        and bool(last_relevant_indices)
        and is_result_expansion_followup(question)
    )


def filter_reused_indices_for_question(question: str, candidate_indices, repo_state, logger):
    q = normalize_question_for_retrieval(question)
    if not q:
        return list(candidate_indices)

    # 只使用当前问题中已经出现过的焦点词
    focus_terms = extract_strong_terms_from_question(q)

    if not focus_terms:
        return list(candidate_indices)

    scored = []
    for idx in candidate_indices:
        path = repo_state.chunk_paths[idx]
        text = repo_state.chunk_texts[idx] or ""

        score = 0
        for term in focus_terms:
            if term and term in text:
                score += 2
            elif term and term in path:
                score += 1

        # 降权：文件名里带立场词，但正文没有对应支撑
        if "非法" in path and "非法" not in text:
            score -= 2
        if "违法" in path and "违法" not in text:
            score -= 2

        scored.append((score, idx))

    scored.sort(reverse=True, key=lambda x: x[0])

    filtered = [idx for score, idx in scored if score > 0]

    if filtered:
        logger.info(f"🧪 [追问结果细筛] focus_terms={focus_terms}")
        logger.info(f"🧪 [追问结果细筛] 保留 {len(filtered)} / {len(candidate_indices)} 个片段")
        return filtered[:12]

    return list(candidate_indices)

def build_search_query(
    *,
    question: str,
    event,
    flags: dict,
    memory_buffer: list[str],
    last_effective_search_query: str | None,
    last_relevant_indices=None,
    logger,
    ollama_api_url: str,
    ollama_model: str,
) -> tuple[str, str]:
    if flags["skip_retrieval"] or flags["is_inventory_query"]:
        return question, ""

    if should_reuse_previous_results(question, event, last_relevant_indices):
        normalized_question = normalize_question_for_retrieval(question) or question
        logger.info("♻️ [追问复用] 跳过 query rewrite，直接复用上一轮结果")
        return normalized_question, ""

    normalized_question = normalize_question_for_retrieval(question)
    base_query = normalized_question or question

    # 关闭上一轮 query / merged_query 继承，避免把历史词串带进来
    context_anchor = ""

    if getattr(event, "name", "") in {"content_followup", "action_request"}:
        logger.info("🧼 [当前轮检索] 仅使用当前问题，不继承上一轮词串")

    if getattr(event, "merged_query", None):
        logger.info("🧼 [当前轮检索] 忽略 merged_query，仅使用当前问题")

    raw_search_query = rewrite_search_query(
        base_query,
        memory_buffer,
        ollama_api_url,
        ollama_model,
        logger,
    )

    # rewrite 不允许新增当前问题里没有的词
    raw_search_query = keep_only_current_question_terms(
        raw_search_query,
        normalized_question or question,
        logger=logger,
    )

    search_query = merge_rewritten_query_with_strong_terms(
        normalized_question or question,
        raw_search_query,
        logger=logger,
    )

    # merge 后再过滤一次，防止重新混入当前问题之外的词
    search_query = keep_only_current_question_terms(
        search_query,
        normalized_question or question,
        logger=logger,
    )

    if not search_query.strip():
        search_query = normalized_question or question

    logger.info(f"🛡️ [强词保底后]：{search_query}")
    return search_query, context_anchor


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