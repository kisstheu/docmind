from __future__ import annotations

from ai.query_rewriter import rewrite_search_query
from app.chat_text_utils import (
    is_related_record_listing_request,
    keep_only_allowed_terms,
    merge_rewritten_query_with_strong_terms,
    normalize_question_for_retrieval,
)
from app.retrieval_flow.query_anchors import (
    extract_explicit_file_anchors as _extract_explicit_file_anchors,
    extract_selector_anchors as _extract_selector_anchors,
    force_append_anchor_terms as _force_append_anchor_terms,
)
from app.retrieval_flow.query_followup import (
    _build_global_timeline_query,
    _force_company_name_anchor_for_followup,
    _is_global_timeline_request,
    _stabilize_followup_merged_query,
    should_keep_followup_anchor,
    should_reuse_previous_results,
    filter_reused_indices_for_question,
)


def _merge_query_terms(*parts: str) -> str:
    merged_terms: list[str] = []
    seen: set[str] = set()

    for part in parts:
        for token in (part or "").split():
            term = token.strip()
            if not term or term in seen:
                continue
            merged_terms.append(term)
            seen.add(term)

    return " ".join(merged_terms).strip()


def build_search_query(
    *,
    question: str,
    event,
    flags: dict,
    memory_buffer: list[str],
    last_effective_search_query: str | None,
    last_user_question: str | None = None,
    last_answer_type: str | None = None,
    last_result_set_items: list[str] | None = None,
    last_result_set_entity_type: str | None = None,
    last_relevant_indices=None,
    logger,
    ollama_api_url: str,
    ollama_model: str,
) -> tuple[str, str]:
    if flags["skip_retrieval"] or flags["is_inventory_query"]:
        return question, ""

    if should_reuse_previous_results(question, event, last_relevant_indices):
        normalized_question = normalize_question_for_retrieval(question) or question
        logger.info("❤️ [追问复用] 跳过 query rewrite，直接复用上一轮结果")
        return normalized_question, ""

    normalized_question = normalize_question_for_retrieval(question)
    context_anchor = ""
    structured_uses_result_set = False

    event_name = getattr(event, "name", "")
    if event_name in {"result_set_followup", "result_set_expansion_followup"}:
        from app.dialog.state_machine import build_result_set_followup_query

        base_query = build_result_set_followup_query(
            question=normalized_question or question,
            last_user_question=last_user_question,
            last_answer_type=last_answer_type,
            last_result_set_items=last_result_set_items,
            last_result_set_entity_type=last_result_set_entity_type,
        )
        logger.info(f"🔆 [结果集追问拼接] {base_query}")
    elif event_name in {"structured_request", "structured_skill_summary"} and last_result_set_items and last_result_set_entity_type:
        from app.dialog.state_machine import build_result_set_followup_query

        base_query = build_result_set_followup_query(
            question=normalized_question or question,
            last_user_question=last_user_question,
            last_answer_type=last_answer_type,
            last_result_set_items=last_result_set_items,
            last_result_set_entity_type=last_result_set_entity_type,
        )
        structured_uses_result_set = True
        logger.info(f"🔆 [结构化继承结果集] {base_query}")
    elif event_name == "structured_request" and getattr(event, "merged_query", None):
        merged_query = normalize_question_for_retrieval(event.merged_query) or str(event.merged_query).strip()
        if should_keep_followup_anchor(question):
            base_query = _merge_query_terms(
                normalize_question_for_retrieval(last_effective_search_query or ""),
                merged_query,
            ) or merged_query or normalized_question or question
            logger.info(f"[structured_request followup] {base_query}")
        else:
            base_query = merged_query or normalized_question or question
            logger.info(f"[structured_request merged] {base_query}")
    else:
        if event_name in {"content_followup", "action_request", "judgment_request", "query_correction"}:
            if getattr(event, "merged_query", None) and should_keep_followup_anchor(question):
                base_query = _stabilize_followup_merged_query(
                    merged_query=event.merged_query,
                    question=question,
                    last_effective_search_query=last_effective_search_query,
                    logger=logger,
                )
                logger.info(f"🔆 [追问继承拼接] {base_query}")
            else:
                base_query = normalized_question or question
                logger.info("🧠 [当前轮检索] 当前问题足够具体，仅使用当前问题")
        else:
            base_query = normalized_question or question

    explicit_file_anchors = _extract_explicit_file_anchors(base_query)
    selector_anchors = _extract_selector_anchors(base_query)
    combined_anchors = list(dict.fromkeys([*explicit_file_anchors, *selector_anchors]))

    if event_name in {"result_set_followup", "result_set_expansion_followup"} or structured_uses_result_set:
        search_query = _force_append_anchor_terms(base_query.strip(), combined_anchors, logger=logger)
        if structured_uses_result_set:
            logger.info("🛝 [结构化结果集] 跳过 rewrite 与新增词过滤，直接使用候选集合查询")
        else:
            logger.info("🛝 [结果集追问] 跳过 rewrite 与新增词过滤，直接使用候选集合查询")
        logger.info(f"🛝 [强词保底后]：{search_query}")
        return search_query, context_anchor

    raw_search_query = rewrite_search_query(
        base_query,
        memory_buffer,
        ollama_api_url,
        ollama_model,
        logger,
    )

    if event_name == "structured_request" and _is_global_timeline_request(question):
        search_query = _build_global_timeline_query()
        search_query = _force_append_anchor_terms(search_query, combined_anchors, logger=logger)
        logger.info("🛝 [全局时间线稳态检索] 使用固定主题词，避免改写抖动")
        logger.info(f"🛝 [强词保底后]：{search_query}")
        return search_query, context_anchor

    if is_related_record_listing_request(question):
        search_query = raw_search_query.strip() or base_query.strip()
        search_query = _force_company_name_anchor_for_followup(
            search_query=search_query,
            base_query=base_query,
            question=question,
            last_answer_type=last_answer_type,
            logger=logger,
        )
        search_query = _force_append_anchor_terms(search_query, combined_anchors, logger=logger)
        logger.info("🛝 [相关记录保词] 跳过新增词过滤，保留规则扩展词")
        logger.info(f"🛝 [强词保底后]：{search_query}")
        return search_query, context_anchor

    if event_name in {"structured_request", "structured_skill_summary"} and any(k in question for k in ["时间线", "按时间", "顺序"]):
        search_query = raw_search_query.strip() or base_query.strip()
        search_query = _force_company_name_anchor_for_followup(
            search_query=search_query,
            base_query=base_query,
            question=question,
            last_answer_type=last_answer_type,
            logger=logger,
        )
        search_query = _force_append_anchor_terms(search_query, combined_anchors, logger=logger)
        logger.debug(f"🔧 [结构化请求原始检索词] {raw_search_query}")
        logger.info("🛝 [结构化请求保锚] 跳过新增词过滤，保留主题补全词")
        logger.info(f"🛝 [强词保底后]：{search_query}")
        return search_query, context_anchor

    if event_name in {"judgment_request", "content_followup", "action_request", "structured_request", "structured_skill_summary"} and should_keep_followup_anchor(question):
        base_query_text = (base_query or "").strip()
        rewritten_text = (raw_search_query or "").strip()

        search_query = _merge_query_terms(base_query_text, rewritten_text) or base_query_text or rewritten_text

        search_query = _force_company_name_anchor_for_followup(
            search_query=search_query,
            base_query=base_query,
            question=question,
            last_answer_type=last_answer_type,
            logger=logger,
        )
        search_query = _force_append_anchor_terms(search_query, combined_anchors, logger=logger)
        logger.info("🛝 [弱追问保锚] 跳过新增词过滤，保留主题补全词")
        logger.info(f"🛝 [强词保底后]：{search_query}")
        return search_query, context_anchor

    allowed_question = base_query or normalized_question or question
    raw_search_query = keep_only_allowed_terms(
        raw_search_query,
        allowed_question,
        logger=logger,
    )

    search_query = merge_rewritten_query_with_strong_terms(
        allowed_question,
        raw_search_query,
        logger=logger,
    )
    search_query = keep_only_allowed_terms(
        search_query,
        allowed_question,
        logger=logger,
    )

    if not search_query.strip():
        search_query = allowed_question

    search_query = _force_company_name_anchor_for_followup(
        search_query=search_query,
        base_query=base_query,
        question=question,
        last_answer_type=last_answer_type,
        logger=logger,
    )
    search_query = _force_append_anchor_terms(search_query, combined_anchors, logger=logger)
    logger.info(f"🛝 [强词保底后]：{search_query}")
    return search_query, context_anchor
