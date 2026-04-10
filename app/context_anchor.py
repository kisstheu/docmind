from __future__ import annotations

import re
from typing import Optional

ANCHOR_STOPWORDS = {
    "帮我", "给我", "请你", "麻烦你", "我想", "我先", "先",
    "分析", "分析下", "分析一下",
    "整理", "整理下", "整理一下",
    "梳理", "梳理下", "梳理一下",
    "总结", "总结下", "总结一下",
    "看看", "看下", "看一下",
    "说说", "讲讲", "聊聊", "研究", "研究下", "研究一下",
    "一下", "吧", "吗", "呢", "呀", "啊",
}


ANCHOR_WEAK_TERMS = {
    "事情", "情况", "问题", "内容", "动作", "方面", "东西",
    "这个", "那个", "这些", "那些", "这样", "这种",
}

GROUP_REFERENCE_MARKERS = {
    "这些",
    "那些",
    "其中",
    "这类",
    "那类",
    "该类",
    "这批",
    "那批",
    "上述",
}


def _extract_candidate_terms(text: str) -> list[str]:
    if not text:
        return []

    candidates: list[str] = []
    candidates.extend(re.findall(r"\d{4}年\d{1,2}月\d{1,2}日", text))
    candidates.extend(re.findall(r"\d{1,2}月\d{1,2}日", text))
    candidates.extend(re.findall(r"[\u4e00-\u9fa5]{2,}", text))
    candidates.extend(re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", text))

    result: list[str] = []
    for token in candidates:
        t = token.strip()
        if not t or t in result:
            continue
        result.append(t)
    return result


def extract_topic_anchor(text: str) -> str:
    """从上一轮有效检索词里抽取尽量干净的主题锚点，不直接复用整句。"""
    if not text:
        return ""

    anchor_terms: list[str] = []
    for token in _extract_candidate_terms(text):
        if token in ANCHOR_STOPWORDS or token in ANCHOR_WEAK_TERMS:
            continue
        if len(token) > 12:
            continue
        if token not in anchor_terms:
            anchor_terms.append(token)

    return " ".join(anchor_terms[:5])


def is_context_dependent_question(question: str, last_query: Optional[str] = None) -> bool:
    q = (question or "").strip()
    if not q:
        return False

    # 很短，通常语义残缺
    if len(q) <= 10:
        return True

    # 带明显承接感的指代/评价语气，且句子不长
    referential_markers = ["他", "她", "对方", "公司", "这个", "那个", "这样", "这种"]
    evaluative_markers = ["合法", "合规", "是否", "算不算", "性质", "合理", "过分", "离谱"]
    follow_markers = ["之后", "后来", "后", "接着", "下一步", "后续"]

    if any(x in q for x in follow_markers):
        return True

    if any(x in q for x in evaluative_markers) and len(q) <= 40:
        return True

    if any(x in q for x in referential_markers) and len(q) <= 30:
        return True

    if any(x in q for x in GROUP_REFERENCE_MARKERS) and len(q) <= 48:
        return True

    # 当前句没有多少可检索实体，但上一轮存在有效主题时，优先视为承接追问
    meaningful_terms = [t for t in _extract_candidate_terms(q) if t not in ANCHOR_STOPWORDS and t not in ANCHOR_WEAK_TERMS]
    if last_query and len(q) <= 30 and len(meaningful_terms) <= 2:
        return True

    return False
