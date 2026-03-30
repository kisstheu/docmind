from __future__ import annotations

import re

from ai.query_rewriter import rewrite_search_query
from app.chat_text_utils import (
    extract_strong_terms_from_question,
    is_related_record_listing_request,
    is_result_expansion_followup,
    keep_only_allowed_terms,
    merge_rewritten_query_with_strong_terms,
    normalize_question_for_retrieval,
)


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


GENERIC_FOLLOWUP_MARKERS = (
    "更多",
    "继续",
    "还有",
    "再来",
    "补充",
    "接着",
    "然后",
    "后续",
)


def _is_generic_followup_question(question: str) -> bool:
    q = normalize_question_for_retrieval(question)
    if not q:
        return False

    if q in {"更多", "继续", "还有", "再来", "补充"}:
        return True

    if len(q) <= 6 and any(marker in q for marker in GENERIC_FOLLOWUP_MARKERS):
        return True

    return False


def _collapse_duplicate_tail_tokens(merged_query: str, question: str) -> str:
    merged = (merged_query or "").strip()
    q = (question or "").strip()
    if not merged or not q:
        return merged

    merged_tokens = [x for x in merged.split() if x.strip()]
    if len(merged_tokens) < 2:
        return merged

    q_norm = normalize_question_for_retrieval(q)
    if not q_norm:
        return merged

    if (
        normalize_question_for_retrieval(merged_tokens[-1]) == q_norm
        and normalize_question_for_retrieval(merged_tokens[-2]) == q_norm
    ):
        return " ".join(merged_tokens[:-1]).strip()

    return merged


def _stabilize_followup_merged_query(
    *,
    merged_query: str,
    question: str,
    last_effective_search_query: str | None,
    logger=None,
) -> str:
    merged = (merged_query or "").strip()
    q = (question or "").strip()
    if not merged:
        return merged

    merged = _collapse_duplicate_tail_tokens(merged, q)

    if not _is_generic_followup_question(q):
        return merged

    last_query = normalize_question_for_retrieval(last_effective_search_query or "")
    if not last_query:
        return merged

    q_norm = normalize_question_for_retrieval(q)
    if q_norm and last_query.endswith(q_norm):
        stabilized = last_query
    else:
        stabilized = f"{last_query} {q_norm}".strip() if q_norm else last_query

    if logger and stabilized != merged:
        logger.info(f"🧷 [追问继承修正] {merged} -> {stabilized}")
    return stabilized


def _extract_explicit_file_anchors(text: str) -> list[str]:
    q = (text or "").strip()
    if not q:
        return []

    pattern = re.compile(
        r"([A-Za-z0-9_\-\u4e00-\u9fa5\s]+?\.(?:txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx|png|jpg|jpeg|bmp|webp))",
        flags=re.IGNORECASE,
    )
    anchors: list[str] = []
    for match in pattern.findall(q):
        raw = re.sub(r"\s+", "", match.strip())
        raw = re.sub(r"^(给我|帮我|请|麻烦你|看下|看看|看一下|查看下|查看|打开|读下|读一下|展示下)+", "", raw)
        raw = raw.strip("，。！？；：,!?;:")
        if not raw:
            continue

        stem = re.sub(r"\.(txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx|png|jpg|jpeg|bmp|webp)$", "", raw, flags=re.IGNORECASE)
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


def _force_company_name_anchor_for_followup(
    *,
    search_query: str,
    base_query: str,
    question: str,
    last_answer_type: str | None,
    logger=None,
) -> str:
    q = (search_query or "").strip()
    if not q:
        q = (base_query or "").strip()

    if last_answer_type != "enumeration_company":
        return q
    if not _is_generic_followup_question(question):
        return q

    combined = f"{q} {base_query or ''}".replace(" ", "")
    has_company_signal = any(
        token in combined
        for token in (
            "\u516c\u53f8",      # 公司
            "\u4f01\u4e1a",      # 企业
            "\u5355\u4f4d",      # 单位
            "\u7ec4\u7ec7",      # 组织
        )
    )
    if not has_company_signal:
        return q

    terms = [x for x in q.split() if x.strip()]
    changed = False
    for anchor in ("\u516c\u53f8", "\u540d\u79f0"):  # 公司, 名称
        if anchor not in terms:
            terms.append(anchor)
            changed = True

    result = " ".join(terms).strip()
    if changed and logger:
        logger.info(f"🧷 [公司名追问保锚] {q} -> {result}")
    return result


def _is_global_timeline_request(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False

    has_timeline_intent = any(x in q for x in ["时间线", "按时间", "按日期", "按时间顺序"])
    has_global_scope = any(x in q for x in ["所有", "全部", "最近所有", "整体", "全局", "总体"])
    return has_timeline_intent and has_global_scope


def _build_global_timeline_query() -> str:
    # 固定主题词，避免 structured_request 依赖模型改写导致召回抖动。
    terms = [
        "最近",
        "时间线",
        "记录",
        "会议纪要",
        "学习笔记",
        "周报",
        "复盘",
        "项目计划",
        "任务看板",
        "访谈",
        "生活",
    ]
    return " ".join(terms)


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
    if event_name in {"result_set_followup", "result_set_expansion_followup"}:
        from app.dialog.state_machine import build_result_set_followup_query

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
                base_query = _stabilize_followup_merged_query(
                    merged_query=event.merged_query,
                    question=question,
                    last_effective_search_query=last_effective_search_query,
                    logger=logger,
                )
                logger.info(f"🔗 [追问继承拼接] {base_query}")
            else:
                base_query = normalized_question or question
                logger.info("🧼 [当前轮检索] 当前问题足够具体，仅使用当前问题")
        else:
            base_query = normalized_question or question

    explicit_file_anchors = _extract_explicit_file_anchors(base_query)

    # 结果集追问：直接使用受控拼接后的查询，不再经过普通 rewrite/过滤链路
    if event_name in {"result_set_followup", "result_set_expansion_followup"}:
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

    if event_name == "structured_request" and _is_global_timeline_request(question):
        search_query = _build_global_timeline_query()
        search_query = _force_append_anchor_terms(search_query, explicit_file_anchors, logger=logger)
        logger.info("🛡️ [全局时间线稳态检索] 使用固定主题词，避免改写抖动")
        logger.info(f"🛡️ [强词保底后]：{search_query}")
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
        search_query = _force_append_anchor_terms(search_query, explicit_file_anchors, logger=logger)
        logger.info("🛡️ [相关记录保词] 跳过新增词过滤，保留规则扩展词")
        logger.info(f"🛡️ [强词保底后]：{search_query}")
        return search_query, context_anchor

    # 结构化请求：不要把补全出来的主题词再过滤掉
    if event_name == "structured_request" and any(k in question for k in ["时间线", "按时间", "顺序"]):
        search_query = raw_search_query.strip() or base_query.strip()
        search_query = _force_company_name_anchor_for_followup(
            search_query=search_query,
            base_query=base_query,
            question=question,
            last_answer_type=last_answer_type,
            logger=logger,
        )
        search_query = _force_append_anchor_terms(search_query, explicit_file_anchors, logger=logger)
        logger.debug(f"🧩 [结构化请求原始检索词] {raw_search_query}")
        logger.info("🛡️ [结构化请求保锚] 跳过新增词过滤，保留主题补全词")
        logger.info(f"🛡️ [强词保底后]：{search_query}")
        return search_query, context_anchor

    # 弱追问 / 判断追问：不要把补全出来的主题词再过滤掉
    if event_name in {"judgment_request", "content_followup", "action_request"} and should_keep_followup_anchor(
            question):
        search_query = raw_search_query.strip() or base_query.strip()
        search_query = _force_company_name_anchor_for_followup(
            search_query=search_query,
            base_query=base_query,
            question=question,
            last_answer_type=last_answer_type,
            logger=logger,
        )
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

    search_query = _force_company_name_anchor_for_followup(
        search_query=search_query,
        base_query=base_query,
        question=question,
        last_answer_type=last_answer_type,
        logger=logger,
    )
    search_query = _force_append_anchor_terms(search_query, explicit_file_anchors, logger=logger)
    logger.info(f"🛡️ [强词保底后]：{search_query}")
    return search_query, context_anchor
