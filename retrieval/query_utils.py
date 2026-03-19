from __future__ import annotations

import re
from typing import Optional, Tuple

EXTENSION_TERMS = {"pdf", "docx", "txt", "md"}
SUPPORTED_EXT = {".txt", ".md", ".docx", ".pdf"}


def extract_company_candidates(text: str):
    candidates = []
    patterns = [
        r'[\u4e00-\u9fa5A-Za-z0-9·（）()\-]{2,}股份有限公司',
        r'[\u4e00-\u9fa5A-Za-z0-9·（）()\-]{2,}有限公司',
        r'[\u4e00-\u9fa5A-Za-z0-9·（）()\-]{2,}集团',
        r'[\u4e00-\u9fa5A-Za-z0-9·（）()\-]{2,}公司',
    ]
    for pattern in patterns:
        for match in re.findall(pattern, text):
            name = match.strip("，。；：、（）() \n\t")
            if len(name) >= 2:
                candidates.append(name)

    result = []
    seen = set()
    for name in candidates:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def classify_org_candidate(name: str):
    name = name.strip()
    if name.endswith("股份有限公司"):
        return "explicit"
    if name.endswith("有限公司"):
        return "explicit"
    if name.endswith("集团") and len(name) >= 4:
        return "explicit"

    generic_names = {
        "公司", "贵公司", "该公司", "本公司", "原公司",
        "大公司", "小公司", "对方公司"
    }
    if name in generic_names:
        return "generic"
    if name.endswith("公司") and len(name) <= 6:
        return "generic"
    return "ambiguous"


def extract_query_terms(search_query: str, question: str):
    text = f"{search_query} {question}"
    text = re.sub(r"[^\w\s\u4e00-\u9fa5]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    raw_terms = []
    raw_terms.extend(re.findall(r"[a-zA-Z0-9_]{2,}", text))
    raw_terms.extend(re.findall(r"[\u4e00-\u9fa5]{2,}", text))

    stop_terms = {
        "我想", "想知道", "请问", "一下", "这个", "那个", "这里", "那里",
        "涉及", "涉及了", "提到", "提到了", "多少", "几个", "哪些", "所有",
        "有没有", "是什么", "什么", "怎么", "为什么", "吗", "呢", "呀", "啊",
        "的呢", "过的", "我的", "那就"
    }

    cleaned = []
    for term in raw_terms:
        term = term.strip()
        if len(term) < 2 or term in stop_terms:
            continue
        if "公司" in term and term != "公司":
            cleaned.append("公司")
            continue
        if "人名" in term and term != "人名":
            cleaned.append("人名")
            continue
        if "项目" in term and term != "项目":
            cleaned.append("项目")
            continue
        cleaned.append(term)

    result = []
    for term in cleaned:
        if term not in result:
            result.append(term)
    return result


def detect_inventory_target(question: str) -> Tuple[Optional[str], Optional[str]]:
    q = re.sub(r"[^\w\s\u4e00-\u9fa5]", "", question)
    target_aliases = {
        "company": ["公司", "单位", "组织", "企业"],
        "person": ["人", "人名", "人物", "同事", "员工"],
        "project": ["项目", "系统", "方案"],
        "place": ["地点", "地方", "城市", "位置"],
    }
    for target_type, aliases in target_aliases.items():
        for alias in aliases:
            if alias in q:
                return target_type, alias
    return None, None
