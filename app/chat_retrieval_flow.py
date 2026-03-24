from __future__ import annotations

import re
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
    keep_only_allowed_terms,
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


def should_keep_followup_anchor(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False

    weak_followups = {
        "我能赢吗", "能赢吗", "能不能赢",
        "有胜算吗", "胜算大吗", "赢面大吗", "把握大吗",
        "合理吗", "合法吗", "过分吗",
        "怎么办", "会怎样", "结果呢", "然后呢", "后来呢",
        "为什么", "依据是什么", "有哪些证据",
    }

    if q in weak_followups:
        return True

    # 短句且语义不完整，保留上一轮锚点
    return len(q) <= 12


def _extract_explicit_file_anchors(text: str) -> list[str]:
    q = (text or "").strip()
    if not q:
        return []

    pattern = re.compile(
        r"([A-Za-z0-9_\-\u4e00-\u9fa5\s]+?\.(?:txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx))",
        flags=re.IGNORECASE,
    )
    anchors: list[str] = []
    for match in pattern.findall(q):
        raw = re.sub(r"\s+", "", match.strip())
        raw = re.sub(r"^(给我|帮我|请|麻烦你|看下|看看|看一下|查看下|查看|打开|读下|读一下|展示下)+", "", raw)
        raw = raw.strip("，。！？；：,!?;:")
        if not raw:
            continue

        stem = re.sub(r"\.(txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx)$", "", raw, flags=re.IGNORECASE)
        for token in (raw, stem):
            t = token.strip()
            if t and t not in anchors:
                anchors.append(t)

    return anchors


def _force_append_anchor_terms(query: str, anchors: list[str], logger=None) -> str:
    q = (query or "").strip()
    if not anchors:
        return q

    merged_terms = [x for x in q.split() if x.strip()]
    appended = []
    for anchor in anchors:
        if anchor not in merged_terms:
            merged_terms.append(anchor)
            appended.append(anchor)

    if appended and logger:
        logger.info(f"🧷 [文件锚词回补] {appended}")

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
        logger.info("♻️ [追问复用] 跳过 query rewrite，直接复用上一轮结果")
        return normalized_question, ""

    normalized_question = normalize_question_for_retrieval(question)
    context_anchor = ""
    explicit_file_anchors: list[str] = []

    event_name = getattr(event, "name", "")
    if event_name == "result_set_followup":
        from app.dialog_state_machine import build_result_set_followup_query

        base_query = build_result_set_followup_query(
            question=normalized_question or question,
            last_user_question=last_user_question,
            last_answer_type=last_answer_type,
            last_result_set_items=last_result_set_items,
            last_result_set_entity_type=last_result_set_entity_type,
        )

        logger.info(f"🔗 [结果集追问拼接] {base_query}")
    else:
        if event_name in {"content_followup", "action_request", "judgment_request"}:
            if getattr(event, "merged_query", None) and should_keep_followup_anchor(question):
                base_query = event.merged_query
                logger.info(f"🔗 [追问继承拼接] {base_query}")
            else:
                base_query = normalized_question or question
                logger.info("🧼 [当前轮检索] 当前问题足够具体，仅使用当前问题")
        else:
            base_query = normalized_question or question

    explicit_file_anchors = _extract_explicit_file_anchors(base_query)

    # 结果集追问：直接使用受控拼接后的查询，不再经过普通 rewrite/过滤链路
    if event_name == "result_set_followup":
        search_query = _force_append_anchor_terms(base_query.strip(), explicit_file_anchors, logger=logger)
        logger.info("🛡️ [结果集追问] 跳过 query rewrite 与新增词过滤，直接使用候选集合查询")
        logger.info(f"🛡️ [强词保底后]：{search_query}")
        return search_query, context_anchor

    raw_search_query = rewrite_search_query(
        base_query,
        memory_buffer,
        ollama_api_url,
        ollama_model,
        logger,
    )
    # 结构化请求：不要把补全出来的主题词再过滤掉
    if event_name == "structured_request" and any(k in question for k in ["时间线", "按时间", "顺序"]):
        search_query = raw_search_query.strip() or base_query.strip()
        search_query = _force_append_anchor_terms(search_query, explicit_file_anchors, logger=logger)
        logger.debug(f"🧩 [结构化请求原始检索词] {raw_search_query}")
        logger.info("🛡️ [结构化请求保锚] 跳过新增词过滤，保留主题补全词")
        logger.info(f"🛡️ [强词保底后]：{search_query}")
        return search_query, context_anchor

    # 弱追问 / 判断追问：不要把补全出来的主题词再过滤掉
    if event_name in {"judgment_request", "content_followup", "action_request"} and should_keep_followup_anchor(
            question):
        search_query = raw_search_query.strip() or base_query.strip()
        search_query = _force_append_anchor_terms(search_query, explicit_file_anchors, logger=logger)
        logger.info("🛡️ [弱追问保锚] 跳过新增词过滤，保留主题补全词")
        logger.info(f"🛡️ [强词保底后]：{search_query}")
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

    search_query = _force_append_anchor_terms(search_query, explicit_file_anchors, logger=logger)
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
