from __future__ import annotations

import re

import numpy as np

from retrieval.query_utils import detect_inventory_target


def is_capability_like_query(question: str) -> bool:
    q = question.strip().lower()
    patterns = ["你是谁", "你能做什么", "你可以做什么", "能做什么", "可以做什么", "能干啥", "干啥", "做啥", "能做啥",
        "有什么功能", "有啥功能", "你的功能", "怎么用", "介绍一下", "help", ]
    return any(p in q for p in patterns)


def is_relation_mismatch_query(question: str) -> bool:
    q = (question or "").replace(" ", "")
    patterns = ["不符", "不一致", "不匹配", "对不上", "冲突", "矛盾", "是否一致", "有没有一致", "文件名", "标题", "正文"]
    return any(p in q for p in patterns)


def is_entity_lookup_query(question: str) -> bool:
    q = (question or "").replace(" ", "").lower()
    patterns = [
        "公司名", "公司名称", "企业名称", "单位名称", "组织名称",
        "人名", "姓名", "名字",
        "项目名", "项目名称",
    ]
    return any(p in q for p in patterns)


def is_expansion_followup_question(question: str) -> bool:
    q = (question or "").replace(" ", "").lower()
    if not q:
        return False
    patterns = ["更多", "还有", "另外", "其他", "别的", "继续", "再来", "补充", "还包括"]
    return any(p in q for p in patterns)


def is_entity_name_lookup_context(question: str, search_terms: list[str]) -> bool:
    if is_entity_lookup_query(question):
        return True

    term_set = {t.strip().lower() for t in (search_terms or []) if t and t.strip()}
    if "名称" in term_set:
        if term_set.intersection({"公司", "人名", "姓名", "项目"}):
            return True
    if term_set.intersection({"公司名", "公司名称", "企业名称", "单位名称", "组织名称", "项目名称"}):
        return True

    return False


def is_company_name_lookup_context(question: str, search_terms: list[str]) -> bool:
    q = (question or "").replace(" ", "").lower()
    if any(x in q for x in ["公司名", "公司名称", "企业名称", "单位名称", "组织名称"]):
        return True

    term_set = {t.strip().lower() for t in (search_terms or []) if t and t.strip()}
    if term_set.intersection({"公司", "公司名", "公司名称", "企业", "企业名称", "单位", "单位名称", "组织", "组织名称"}):
        return True
    if "名称" in term_set and term_set.intersection({"公司", "企业", "单位", "组织"}):
        return True
    return False


def is_weak_query(question: str, search_terms: list[str]) -> bool:
    q = question.strip().lower()

    if is_capability_like_query(q):
        return True

    # “查实体名称”类是明确检索需求，不按弱问题处理。
    if is_entity_lookup_query(q):
        return False

    if not search_terms:
        return True

    joined = "".join(search_terms).strip()
    if len(search_terms) == 1 and len(joined) <= 2:
        return True

    weak_terms = {"能做啥", "做啥", "干啥", "功能", "帮助", "help"}
    if any(term in weak_terms for term in search_terms):
        return True

    return False


def should_enable_fallback(search_terms: list[str], weak_query: bool) -> bool:
    if weak_query:
        return False
    return bool(search_terms)


def rescue_entity_lookup_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if scores.size == 0 or top_k <= 0:
        return []

    sorted_idx = [int(i) for i in np.argsort(scores)[::-1]]
    strong_hits = [i for i in sorted_idx if float(scores[i]) > 0.18][:top_k]
    if strong_hits:
        return strong_hits

    return sorted_idx[: min(top_k, 8)]


def determine_query_flags(question: str):
    inventory_triggers = ["多少", "哪些", "有哪些", "提到", "涉及", "所有", "盘点", "过"]
    relationship_queries = ["对我如何", "关系好", "评价", "他人怎么样", "对他"]
    greetings = ["你好", "嗨", "在吗", "谢谢", "好的", "ok", "嗯", "哈哈", "知道了", "原来如此", "厉害", "棒", "牛逼",
                 "多谢", "感谢"]
    system_queries = ["能干啥", "你是谁", "怎么用", "你能做什么", "你的功能", "介绍一下", "能做什么", "可以做什么",
        "你可以做什么", "有什么功能", "有啥功能"]

    inventory_target_type, inventory_target_label = detect_inventory_target(question)
    is_inventory_query = inventory_target_type is not None and any(t in question for t in inventory_triggers)
    is_relationship_query = any(q in question for q in relationship_queries)
    skip_retrieval = False
    q_lower = question.lower().strip()
    # 移除标点符号后的纯文本长度
    pure_text = re.sub(r'[^\w\s]', '', q_lower)

    # 只有当用户输入的内容基本上全是问候语（比如长度极短），或者是精确等于问候语时，才跳过
    is_pure_greeting = any(q_lower == g or pure_text == g for g in greetings)

    if is_pure_greeting and not is_relationship_query:
        skip_retrieval = True
    elif any(word in question for word in system_queries):
        skip_retrieval = True

    return {"inventory_target_type": inventory_target_type, "inventory_target_label": inventory_target_label,
        "is_inventory_query": is_inventory_query, "is_relationship_query": is_relationship_query,
        "skip_retrieval": skip_retrieval, }


def is_related_record_listing_query(question: str) -> bool:
    q = (question or "").replace(" ", "").lower()
    has_related = any(x in q for x in ["有关的记录", "相关记录", "有关记录", "相关的记录", "有关文档", "相关文档"])
    has_listing = any(x in q for x in ["哪些", "哪几", "有哪", "最近"])
    return has_related and has_listing


def is_compare_intent_query(question: str) -> bool:
    q = (question or "").replace(" ", "")
    generic_compare_keywords = ["不同", "区别", "差异", "异同", "比较", "对比", "相同", "一样", "一致"]
    if any(kw in q for kw in generic_compare_keywords):
        return True

    # 公司实体比对常见口语问法，避免漏判导致召回过窄。
    company_compare_patterns = [
        "是不是一家",
        "是否一家",
        "是一个吗",
        "是不是一个",
        "是否一个",
        "是不是同一个",
        "是否同一个",
        "是同一个吗",
        "同一个吗",
        "同一家公司",
        "是不是同一家公司",
        "是否同一家公司",
        "是同一家公司吗",
        "是不是同一个公司",
        "是否同一个公司",
    ]
    return any(p in q for p in company_compare_patterns)


def is_file_location_lookup_query(question: str, search_query: str) -> bool:
    q = re.sub(r"\s+", "", (question or "").lower())
    sq = re.sub(r"\s+", "", (search_query or "").lower())
    merged = q + sq
    if not merged:
        return False

    direct_patterns = [
        "在哪个文件", "在那个文件", "是哪个文件", "是那个文件",
        "哪个文件", "哪些文件", "哪份文件", "文件里", "文件中",
        "在哪个文档", "在那个文档", "是哪个文档", "是那个文档",
        "哪个文档", "哪些文档",
        "在哪个记录", "是哪个记录", "哪个记录", "哪些记录",
    ]
    if any(p in merged for p in direct_patterns):
        return True

    confirmation_patterns = [
        "只有这个吗", "就这个吗", "只有这一个吗", "就这一个吗",
        "只有这些吗", "就这些吗", "只有这几个吗", "就这几个吗",
    ]
    if any(p in q for p in confirmation_patterns) and any(x in sq for x in ["文件", "文档", "记录"]):
        return True

    return False
