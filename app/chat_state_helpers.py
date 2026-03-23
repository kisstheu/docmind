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
    a = (answer_text or "").strip()

    numbered_lines = re.findall(r"(?:^|\n)\s*(?:\d+[.、]|[-*•])\s*", a)

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

    if "文件" in q or "文档" in q:
        if len(numbered_lines) >= 2:
            return "enumeration_file"
        if "以下文件" in a or "以下文档" in a:
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

    for item in extract_numbered_items(answer_text):
        t = item.strip()

        t = re.sub(r"（\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}）$", "", t).strip()

        if t:
            items.append(t)

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

    t = re.sub(r"\s+", "", t)
    return t.strip()


def looks_like_real_company_name(text: str) -> bool:
    t = text.strip()

    # 太短一般不是可靠公司名
    if len(t) < 6:
        return False

    # 过滤“若干中文 + 公司”这种泛指形式
    # 例如：某地公司、某某公司、泛称公司
    if re.fullmatch(r"[\u4e00-\u9fa5]{2,6}公司", t):
        return False

    # 过滤代词/泛指公司
    if re.fullmatch(r"[某该本此这那一][\u4e00-\u9fa5]*公司", t):
        return False

    # 至少包含更强的公司结构特征
    strong_markers = [
        "有限公司",
        "股份有限公司",
        "有限责任公司",
        "集团",
        "科技",
        "实业",
    ]
    return any(marker in t for marker in strong_markers)


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
    else:
        state.last_result_set_items = None
        state.last_result_set_entity_type = None

    return state


def update_state_after_retrieval_answer(state, question: str, answer_text: str, logger):
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
        state.last_result_set_items = company_items
        state.last_result_set_entity_type = "公司"
        raw_items = extract_numbered_items(answer_text)


        for item in raw_items:
            cleaned = normalize_company_item(item)
            if looks_like_real_company_name(cleaned):
                company_items.append(cleaned)

        state.last_result_set_items = company_items

        logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={answer_type}")
        logger.debug(f"🧪 [候选集合提取] raw_items={raw_items}")
        logger.debug(f"🧪 [候选集合提取] company_items={company_items}")

    else:
        logger.debug(f"🧪 [answer_type识别] q={question} | answer_type={answer_type}")

    logger.debug(
        f"💾 [状态写回] "
        f"last_user_question={state.last_user_question} | "
        f"last_content_route={state.last_content_route} | "
        f"last_answer_type={state.last_answer_type}"
    )

    return state