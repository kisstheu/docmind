from __future__ import annotations

import re

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

FOLLOWUP_EVENT_NAMES = {
    "content_followup",
    "result_set_followup",
    "result_set_expansion_followup",
    "action_request",
    "judgment_request",
    "query_correction",
}

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


def update_state_after_local_answer(
    state,
    question: str,
    answer: str,
    route: str,
    local_topic: str | None,
    is_content_answer: bool,
):
    state.last_user_question = question
    state.last_route = route
    state.last_local_topic = local_topic
    state.last_answer_preview = answer[:200]
    state.last_answer_text = answer
    if local_topic in {"category_summary", "category_count_breakdown", "category_overview"}:
        state.last_category_context_answer = answer

    if is_content_answer:
        state.last_content_user_question = question
        state.last_content_route = route
        state.last_content_topic = local_topic

    local_answer_type = infer_local_answer_type(question, answer, local_topic)
    state.last_answer_type = local_answer_type

    if local_answer_type == "enumeration_file":
        file_items = extract_file_items(answer)
        state.last_result_set_items = file_items
        state.last_result_set_entity_type = "文件"

    return state


def update_state_after_retrieval_answer(
    state,
    question: str,
    answer_text: str,
    logger,
    event_name: str | None = None,
):
    prev_result_set_items = list(state.last_result_set_items) if state.last_result_set_items else None
    prev_result_set_entity_type = state.last_result_set_entity_type
    prev_answer_type = state.last_answer_type
    is_followup_turn = _is_followup_turn(question, event_name=event_name)

    state.last_user_question = question
    state.last_route = "normal_retrieval"
    state.last_local_topic = None

    state.last_content_user_question = question
    state.last_content_route = "normal_retrieval"
    state.last_content_topic = None

    state.last_answer_text = answer_text
    state.last_answer_preview = answer_text[:200]

    answer_type = infer_answer_type(question, answer_text)
    state.last_answer_type = answer_type

    if answer_type == "enumeration_company":
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

        logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={answer_type}")
        logger.debug(f"🧪 [候选集合提取] raw_items={raw_items}")
        logger.debug(f"🧪 [候选集合提取] company_items={company_items}")
    elif answer_type == "enumeration_file":
        file_items = extract_file_items(answer_text)

        if prev_result_set_entity_type == "文件" and prev_result_set_items:
            if file_items and is_followup_turn:
                file_items = _merge_result_set_items(prev_result_set_items, file_items)
                logger.debug("🧪 [候选集合提取] 追问场景合并文件候选集合")
            elif not file_items and _contains_no_new_signal(answer_text):
                file_items = prev_result_set_items
                logger.debug("🧪 [候选集合提取] 本轮无新增文件，沿用上一轮文件候选集合")

        state.last_result_set_items = file_items
        state.last_result_set_entity_type = "文件"

        logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={answer_type}")
        logger.debug(f"🧪 [候选集合提取] file_items={file_items}")
    elif answer_type == "enumeration_person":
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
        preserve_source_file_refs = (
            bool(fallback_file_items)
            and _looks_like_source_backed_analytic_answer(question, answer_text)
        )
        fallback_to_file_result_set = (
            bool(fallback_file_items)
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
            and prev_result_set_entity_type in entity_to_answer_type
            and bool(prev_result_set_items)
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
            logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={state.last_answer_type}")
            logger.debug(f"🧪 [候选集合提取] file_items={fallback_file_items}")
        elif preserve_source_file_refs:
            if prev_result_set_entity_type is None and prev_result_set_items and is_followup_turn:
                fallback_file_items = _merge_result_set_items(prev_result_set_items, fallback_file_items)
                logger.debug("🧪 [状态保留] 分析回答附带来源文件，合并上一轮来源文件候选集合")

            state.last_result_set_items = fallback_file_items
            state.last_result_set_entity_type = None
            state.last_answer_type = None
            logger.debug("🧪 [状态保留] 分析回答仅保留来源文件候选，不视为文件结果集")
            logger.debug(f"🧪 [候选集合提取] analytic_source_file_items={fallback_file_items}")
        elif keep_result_set_context or preserve_result_set_on_result_set_followup or preserve_file_result_set_on_no_evidence_followup:
            state.last_result_set_items = prev_result_set_items
            state.last_result_set_entity_type = prev_result_set_entity_type
            state.last_answer_type = prev_answer_type or entity_to_answer_type.get(prev_result_set_entity_type)
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
            logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={answer_type}")

    logger.debug(
        f"🧠 [状态写回] "
        f"last_user_question={state.last_user_question} | "
        f"last_content_route={state.last_content_route} | "
        f"last_answer_type={state.last_answer_type}"
    )

    return state
