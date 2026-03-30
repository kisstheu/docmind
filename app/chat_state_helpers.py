from __future__ import annotations

import re


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


def infer_answer_type(user_question: str, answer_text: str) -> str | None:
    q = (user_question or "").strip()
    q_norm = re.sub(r"[？?！!，,。.\s]+", "", q)
    a = (answer_text or "").strip()

    numbered_lines = re.findall(r"(?:^|\n)\s*(?:\d+[.、]|[-*•])\s*", a)

    if any(x in q for x in ["公司名", "公司名称", "企业名称", "单位名称", "组织名称"]):
        if len(numbered_lines) >= 2:
            return "enumeration_company"

    if "公司" in q and any(x in q for x in ["多少", "哪些", "哪几家", "哪几个", "列出", "列一下", "提到了"]):
        if len(numbered_lines) >= 2:
            return "enumeration_company"
        if "以下公司" in a:
            return "enumeration_company"
        if "提到了以下公司" in a:
            return "enumeration_company"
        if "总共有" in a and "公司" in a:
            return "enumeration_company"
        if "总共提到了" in a and "公司" in a:
            return "enumeration_company"
        if a.count("有限公司") >= 2:
            return "enumeration_company"

    expansion_markers = ["更多", "还有", "继续", "再来", "补充"]
    if any(x in q_norm for x in expansion_markers):
        if len(numbered_lines) >= 1 and ("公司" in a or "名称" in a):
            return "enumeration_company"

        no_new_patterns = [
            "没有发现其他",
            "没有发现新的",
            "没有其他新的",
            "暂未发现新的",
            "未发现其他",
            "没有找到其他",
            "没有找到新的",
            "未找到新的",
            "没有更多",
        ]
        if ("公司" in a or "名称" in a) and any(p in a for p in no_new_patterns):
            return "enumeration_company"

    if "文件" in q or "文档" in q or "记录" in q:
        if len(numbered_lines) >= 2:
            return "enumeration_file"
        if "以下文件" in a or "以下文档" in a:
            return "enumeration_file"
        if re.search(r"(?:文件|文档|记录)[【\[][^】\]\n]+[】\]]", a):
            return "enumeration_file"

    if "谁" in q or "哪些人" in q or "人物" in q:
        if len(numbered_lines) >= 2:
            return "enumeration_person"
        if "以下人物" in a or "以下人员" in a:
            return "enumeration_person"

    return None


def extract_numbered_items(answer_text: str) -> list[str]:
    items: list[str] = []

    for line in (answer_text or "").splitlines():
        line = line.strip()
        if not line:
            continue

        # 1. 编号列表：1. xxx / 1、xxx
        m = re.match(r"^\d+[.、]\s*(.+?)\s*$", line)
        if m:
            items.append(m.group(1).strip())
            continue

        # 2. 项目符号列表：* xxx / - xxx / • xxx
        m = re.match(r"^[\*\-•]\s+(.+?)\s*$", line)
        if m:
            items.append(m.group(1).strip())
            continue

    return items

def extract_file_items(answer_text: str) -> list[str]:
    items: list[str] = []

    def _add_file_item(raw_item: str) -> None:
        t = (raw_item or "").strip()
        if not t:
            return

        t = re.sub(r"（\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}）$", "", t).strip()
        t = re.sub(r"[。；;，,]+$", "", t).strip()
        if not t:
            return
        if t not in items:
            items.append(t)

    for item in extract_numbered_items(answer_text):
        _add_file_item(item)

    for m in re.findall(r"(?:文件|文档|记录)[【\[]([^】\]\n]+)[】\]]", answer_text or ""):
        _add_file_item(m)

    return items


def infer_local_answer_type(user_question: str, answer_text: str, local_topic: str | None) -> str | None:
    q = (user_question or "").strip()
    a = (answer_text or "").strip()
    topic = (local_topic or "").strip()

    if topic == "time" and ("文件" in q or "文档" in q):
        items = extract_file_items(a)
        if len(items) >= 1:
            return "enumeration_file"

    if topic in {"list_files", "list_files_by_topic"}:
        items = extract_file_items(a)
        if len(items) >= 1:
            return "enumeration_file"

    return None

def normalize_company_item(text: str) -> str:
    t = text.strip()

    # 只去掉“（全称未给出）”“（指代...）”这种解释性附注
    t = re.sub(r"（(?:全称未给出|指代.*?|未给出.*?|名称未明确.*?)）", "", t)
    t = re.sub(r"\((?:full name not given|alias|placeholder.*?)\)", "", t, flags=re.I)

    # 模型输出可能会带截断省略号，避免污染候选集合
    t = re.sub(r"(?:\.\.\.|…)+$", "", t).strip()

    # 英文公司名需要保留词边界，避免被误判
    t = re.sub(r"\s+", " ", t).strip()
    return t.strip()


GENERIC_COMPANY_REFERENCES = {
    "公司",
    "该公司",
    "本公司",
    "某公司",
    "对方公司",
    "企业",
    "单位",
    "组织",
    "公司名",
    "公司名称",
    "企业名称",
    "单位名称",
    "组织名称",
    "名称",
    "合作伙伴",
    "合作方",
    "甲方",
    "乙方",
}


def is_generic_company_reference(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    t_lower = t.lower()

    if t in GENERIC_COMPANY_REFERENCES or t_lower in GENERIC_COMPANY_REFERENCES:
        return True

    # 模糊指代公司
    if re.fullmatch(r"[某该本此这那一]\S*公司", t):
        return True
    if re.fullmatch(r"[\u4e00-\u9fa5]{1,6}公司", t) and len(t) <= 4:
        return True

    # 明显不是实体名而是描述词
    bad_fragments = ["参考片段", "根据", "找到", "更多", "如下", "以下", "没有找到", "未找到"]
    if any(x in t for x in bad_fragments):
        return True

    return False


def looks_like_real_company_name(text: str) -> bool:
    t = text.strip()
    t_lower = t.lower()
    if is_generic_company_reference(t):
        return False

    # 太短一般不是可靠公司名
    if len(t) < 3:
        return False

    # 过滤“若干中文 + 公司”这种泛指形式
    # 例如：某地公司、某某公司、泛称公司
    if re.fullmatch(r"[\u4e00-\u9fa5]{2,6}公司", t):
        return False

    # 过滤代词/泛指公司
    if re.fullmatch(r"[某该本此这那一][\u4e00-\u9fa5]*公司", t):
        return False

    # 至少包含更强的中文公司结构特征
    strong_markers = [
        "有限公司",
        "股份有限公司",
        "有限责任公司",
        "集团",
        "科技",
        "实业",
    ]
    if any(marker in t for marker in strong_markers):
        return True

    # 英文公司名常见后缀/组织词
    if re.search(
        r"\b(co|company|inc|ltd|corp|group|advisors?|solutions?|global|holdings?|technology|tech)\b",
        t_lower,
    ):
        return True

    # 兜底：至少两个英文词且长度足够，避免把普通短词当公司名
    if re.search(r"[a-z]", t_lower):
        parts = [x for x in re.split(r"[^a-z0-9']+", t_lower) if x]
        if len(parts) >= 2 and len("".join(parts)) >= 8:
            return True

    # 中文简称/品牌名（无“公司”后缀）兜底保留，提高候选召回
    if re.fullmatch(r"[\u4e00-\u9fa5]{3,12}", t):
        return True

    if re.fullmatch(r"[\u4e00-\u9fa5]{2,12}(股份|科贸|技术|科技|生物|视觉|创想|新博通)$", t):
        return True

    return False


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


def update_state_after_retrieval_answer(state, question: str, answer_text: str, logger):
    prev_result_set_items = list(state.last_result_set_items) if state.last_result_set_items else None
    prev_result_set_entity_type = state.last_result_set_entity_type
    prev_answer_type = state.last_answer_type

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
        if not company_items and prev_result_set_entity_type == "公司" and prev_result_set_items:
            company_items = prev_result_set_items
            logger.debug("🧷 [候选集合提取] 本轮无新增公司，沿用上一轮公司候选集合")

        state.last_result_set_items = company_items
        state.last_result_set_entity_type = "公司"
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
            # 兜底：只要不是泛称指代就先保留，避免误删简称
            company_items.append(cleaned)

        # 回退：若严格规则全被过滤，则至少保留可读的原始候选，避免结果集状态丢失
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

        state.last_result_set_items = company_items

        logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={answer_type}")
        logger.debug(f"🧪 [候选集合提取] raw_items={raw_items}")
        logger.debug(f"🧪 [候选集合提取] company_items={company_items}")
    elif answer_type == "enumeration_file":
        file_items = extract_file_items(answer_text)
        state.last_result_set_items = file_items
        state.last_result_set_entity_type = "文件"

        logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={answer_type}")
        logger.debug(f"🧪 [候选集合提取] file_items={file_items}")
    elif answer_type == "enumeration_person":
        person_items = extract_numbered_items(answer_text)
        state.last_result_set_items = person_items
        state.last_result_set_entity_type = "人物"

        logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={answer_type}")
        logger.debug(f"🧪 [候选集合提取] person_items={person_items}")

    else:
        q_norm = re.sub(r"[？?！!，,。.\s]+", "", (question or ""))
        keep_company_context = (
            prev_result_set_entity_type == "公司"
            and bool(prev_result_set_items)
            and any(x in q_norm for x in ["更多", "还有", "继续", "再来", "补充"])
            and any(p in (answer_text or "") for p in ["没有发现", "未发现", "没有找到", "未找到", "没有更多"])
        )

        if keep_company_context:
            state.last_result_set_items = prev_result_set_items
            state.last_result_set_entity_type = "公司"
            state.last_answer_type = prev_answer_type or "enumeration_company"
            logger.debug("🧷 [状态保留] 扩展追问未新增公司，保留上一轮公司候选集合")
            logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={state.last_answer_type}")
        else:
            state.last_result_set_items = None
            state.last_result_set_entity_type = None
            logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={answer_type}")

    logger.debug(
        f"💾 [状态写回] "
        f"last_user_question={state.last_user_question} | "
        f"last_content_route={state.last_content_route} | "
        f"last_answer_type={state.last_answer_type}"
    )

    return state
