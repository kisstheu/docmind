from __future__ import annotations

import re


def extract_result_set_from_answer(answer: str, entity_type: str = "文件") -> tuple[list[str], str]:
    items = re.findall(r"\d+\.\s*([^（]+?)\s*（", answer)
    items = [item.strip() for item in items if item.strip()]
    return items, entity_type


RESULT_SET_FOLLOWUP_PATTERNS = [
    r"^哪些是",
    r"^哪个是",
    r"^是哪个文件",
    r"^是哪个文档",
    r"^是哪个记录",
    r"^是在哪个文件",
    r"^是在哪个文档",
    r"^是在哪个记录",
    r"^在哪个文件",
    r"^在哪个文档",
    r"^在哪个记录",
    r"^在哪份文件",
    r"^在哪份文档",
    r"^在哪份记录",
    r"^哪几个是",
    r"^哪些属于",
    r"^哪些不是",
    r"^哪个不是",
    r"^其中哪些",
    r"^这里面哪些",
    r"^其中哪个",
    r"^这里面哪个",
    r"^哪些提到",
    r"^哪些涉及",
    r"^哪些和.*有关",
    r"^哪些与.*有关",
    r"^哪个最早",
    r"^哪个最新",
    r"^哪个最晚",
    r"^都有哪些",
    r"^有哪些",
    r"^分别有哪些",
    r"^分别是哪些",
    r"^具体有哪些",
    r"^具体是哪些",
    r"^都是什么",
    r"^分别是什么",
    r"^分别是关于什么",
    r"^分别是关于什么的",
    r"^各自是关于什么",
    r"^各自讲了什么",
    r"^分别讲了什么",
    r"^分别在说什么",
    r"^列一下",
    r"^展开说",
    r"^还有别的",
    r"^还有哪些",
    r"^哪几个",
    r"^哪几家",
    r"^目前知道的是哪几个",
    r"^目前知道的是哪几家",
    r"^所以.*哪几个",
    r"^所以.*哪几家",
    r"^这两个",
    r"^这两个是",
    r"^这两个其实",
    r"^是不是说这两个",
    r"^这两个.*相同",
    r"^这两个.*一致",
    r"^它们是",
    r"^它们.*相同",
    r"^它们.*一致",
    r"^也就是说",
    r"^也就是说其实",
    r"^其实是",
    r"^其实.*一样",
    r"^其实.*相同",
    r"^其实.*一致",
    r"^也就是说.*相同",
    r"^也就是说.*一致",
    r"^也就是说.*一样",
]
RESULT_SET_CONTINUATION_PATTERNS = [
    r"^更多$",
    r"^继续$",
    r"^再来$",
    r"^补充$",
    r"^还有吗",
    r"^还有没有",
    r"^还有别的",
    r"^还有其他",
    r"^还有哪些",
    r"^还有别的吗",
    r"^还有其他的吗",
    r"^还包括",
    r"^除此之外",
    r"^另外还有",
]


RESULT_SET_COMPARISON_TERMS = [
    "不同", "区别", "差异", "异同",
    "相同", "一样", "一致",
    "对比", "比较",
]


RESULT_SET_GROUP_REF_TERMS = [
    "这几个", "这几份", "这两个", "这三", "这两",
    "这些", "它们", "上述", "前面", "上面", "其中",
]


def looks_like_result_set_followup(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False

    return (
        any(re.search(p, q) for p in RESULT_SET_FOLLOWUP_PATTERNS)
        or any(re.search(p, q) for p in RESULT_SET_CONTINUATION_PATTERNS)
    )


def looks_like_result_set_continuation_followup(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    return any(re.search(p, q) for p in RESULT_SET_CONTINUATION_PATTERNS)


def looks_like_result_set_comparison_followup(question: str) -> bool:
    q = re.sub(r"\s+", "", (question or "").lower())
    if not q:
        return False

    if not any(term in q for term in RESULT_SET_COMPARISON_TERMS):
        return False

    if any(term in q for term in RESULT_SET_GROUP_REF_TERMS):
        return True

    return bool(re.search(r"(?:[一二两三四五六七八九十\d]+)(?:个|份|项|条)", q))


def last_turn_looks_like_enumeration(last_answer: str | None) -> bool:
    if not last_answer:
        return False

    text = last_answer.strip()

    numbered_lines = re.findall(r"(?:^|\n)\s*(?:\d+[.、]|[-*•])\s*", text)
    if len(numbered_lines) >= 2:
        return True

    if re.search(r"明确提到了\d+[家个项份条]", text):
        return True

    if re.search(r"提到了以下\d*[家个项份条]?", text):
        return True

    if re.search(r"以下\d+[家个项份条]", text):
        return True

    if "以下公司" in text or "提到了以下公司" in text:
        return True

    if "如下：" in text or "如下:" in text or "分别如下" in text:
        return True

    if text.count("有限公司") >= 2:
        return True

    return False


def infer_result_set_anchor(
    last_user_question: str | None,
    last_answer_type: str | None,
) -> str | None:
    prev = (last_user_question or "").strip()

    if last_answer_type == "enumeration_company":
        return "文档里提到的公司"
    if last_answer_type == "enumeration_file":
        return "上一轮提到的文件"
    if last_answer_type == "enumeration_person":
        return "上一轮提到的人物"

    if not prev:
        return None

    if "公司" in prev:
        return "文档里提到的公司"
    if "文件" in prev or "文档" in prev:
        return "上一轮提到的文件"
    if "人" in prev or "人物" in prev:
        return "上一轮提到的人物"

    return None


FILE_FOLLOWUP_STOP_TERMS = {
    "什么", "为何", "为什么", "怎么", "如何", "是否", "是不是",
    "哪个", "哪些", "哪几个", "几个", "三", "三个", "两", "两个", "几",
    "有啥", "有什么", "内容", "文件", "文档", "资料",
    "不同", "区别", "差异", "异同", "相同", "一样", "一致", "比较", "对比",
}


def _normalize_for_name_match(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "", (text or "").lower())


def _extract_question_focus_terms_for_files(question: str) -> list[str]:
    q = (question or "").lower()
    q = re.sub(r"[^\w\u4e00-\u9fa5]+", " ", q)
    raw_terms = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", q)

    terms: list[str] = []
    for term in raw_terms:
        t = term.strip()
        if not t or t in FILE_FOLLOWUP_STOP_TERMS:
            continue
        if re.fullmatch(r"\d+", t):
            continue
        if t not in terms:
            terms.append(t)
    return terms


def _narrow_result_set_files_by_question(items: list[str], question: str) -> list[str]:
    if not items:
        return items

    question_norm = _normalize_for_name_match(question)
    focus_terms = _extract_question_focus_terms_for_files(question)
    matched: list[str] = []
    for item in items:
        normalized_name = _normalize_for_name_match(item)
        # 1) 问题词直接命中文件名
        if focus_terms and any(term in normalized_name for term in focus_terms):
            matched.append(item)
            continue

        # 2) 反向匹配：文件名里的有效片段是否出现在问题中
        stem = re.sub(r"\.(txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx|png|jpg|jpeg|bmp|webp)$", "", item, flags=re.I)
        raw_parts = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", stem.lower())
        parts: list[str] = []
        for p in raw_parts:
            if p in FILE_FOLLOWUP_STOP_TERMS or re.fullmatch(r"\d+", p):
                continue
            if p not in parts:
                parts.append(p)

        matched_by_part = False
        for part in parts:
            if part in question_norm:
                matched_by_part = True
                break

            if re.fullmatch(r"[\u4e00-\u9fa5]{4,}", part):
                for width in (2, 3):
                    for i in range(0, len(part) - width + 1):
                        sub = part[i:i + width]
                        if sub in FILE_FOLLOWUP_STOP_TERMS:
                            continue
                        if sub in question_norm:
                            matched_by_part = True
                            break
                    if matched_by_part:
                        break
            if matched_by_part:
                break

        if matched_by_part:
            matched.append(item)

    # 至少保留 2 个候选，避免过度收窄
    return matched if len(matched) >= 2 else items


def build_result_set_followup_query(
    question: str,
    last_user_question: str | None,
    last_answer_type: str | None,
    last_result_set_items: list[str] | None = None,
    last_result_set_entity_type: str | None = None,
) -> str:
    q = (question or "").strip()

    if last_result_set_items:
        entity_type = (last_result_set_entity_type or "项").strip()
        candidate_items = list(last_result_set_items)
        if entity_type == "文件":
            candidate_items = _narrow_result_set_files_by_question(candidate_items, q)
        item_text = "；".join(candidate_items[:20])

        if looks_like_result_set_continuation_followup(q):
            return (
                f"已知{entity_type}集合如下：{item_text}。"
                f"请继续在知识库中检索，判断是否还有其他符合条件的{entity_type}，并避免重复已知项。"
                f"当前追问：{q}"
            )

        if entity_type == "文件":
            return (
                f"已知文件如下：{item_text}。"
                f"请基于这些文件的内容回答：{q}"
            )

        return (
            f"候选{entity_type}如下：{item_text}。"
            f"请只在这些候选项中回答：{q}"
        )

    anchor = infer_result_set_anchor(last_user_question, last_answer_type)

    if not anchor:
        return q

    if q.startswith(("都有哪些", "有哪些", "分别有哪些", "分别是哪些", "具体有哪些", "具体是哪些", "都是什么")):
        return f"{anchor}分别有哪些？请列出完整名单。"

    if q.startswith(("哪些", "哪个", "哪几个")):
        return f"{anchor}中，{q}"

    if q.startswith(("其中", "这里面")):
        return f"{anchor}里，{q}"

    return f"基于{anchor}，回答：{q}"
