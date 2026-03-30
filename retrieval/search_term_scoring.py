from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np

from retrieval.query_utils import (
    classify_org_candidate,
    extract_company_candidates,
)


def is_over_generic_term(term: str) -> bool:
    t = (term or "").strip().lower()

    generic_terms = {"公司", "事情", "情况", "问题", "内容", "资料", "文件", "记录", "信息", "人员", "地方", "时间",
        "东西", "方面", }

    return t in generic_terms


def is_date_like_term(term: str) -> bool:
    t = (term or "").strip().lower()

    # 纯数字，且位数很短，通常是日期/编号噪音
    if re.fullmatch(r"\d{1,4}", t):
        return True

    # 常见中文日期表达
    if re.fullmatch(r"\d{1,2}号", t):
        return True
    if re.fullmatch(r"\d{1,2}日", t):
        return True
    if re.fullmatch(r"\d{1,2}月", t):
        return True
    if re.fullmatch(r"\d{4}年", t):
        return True
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", t):
        return True

    return False


def is_sentence_like_term(term: str) -> bool:
    t = (term or "").strip()
    if not t:
        return False
    if term in {"动作", "情况", "事情"}:
        return True
    # 太长的中文串，大概率不是关键词，而是整句/短语
    if re.fullmatch(r"[\u4e00-\u9fa5]{7,}", t):
        return True

    # 明显带动作语气的短句
    if re.search(r"(给我|帮我|我想|我先|分析|整理|梳理|看下|看看|说说|讲讲)", t):
        return True

    return False


def is_result_set_boilerplate_term(term: str) -> bool:
    t = (term or "").strip()
    if not t:
        return False
    patterns = [
        "已知文件如下",
        "已知文档如下",
        "候选文件如下",
        "候选文档如下",
        "请基于这些文件的内容回答",
        "请基于这些文档的内容回答",
    ]
    return any(p in t for p in patterns)


def should_score_filename_term(term: str) -> bool:
    t = (term or "").strip().lower()
    if not t:
        return False
    if t in {"屏幕", "截图", "屏幕截图", "图片", "图像"}:
        return False
    if is_date_like_term(t):
        return False
    if is_sentence_like_term(t):
        return False
    return True


def build_company_candidate_df(docs: list[str]) -> Tuple[Dict[str, int], int]:
    candidate_df: Dict[str, int] = {}
    total_docs = 0

    for doc_text in docs or []:
        total_docs += 1
        seen_in_doc = set()
        for name in extract_company_candidates(doc_text or ""):
            if classify_org_candidate(name) == "generic":
                continue
            seen_in_doc.add(name)

        for name in seen_in_doc:
            candidate_df[name] = candidate_df.get(name, 0) + 1

    return candidate_df, max(1, total_docs)


def company_hint_bonus(
    chunk_text: str,
    candidate_df: dict[str, int] | None = None,
    total_docs: int = 1,
) -> float:
    text = chunk_text or ""
    if not text:
        return 0.0

    org_hits = extract_company_candidates(text)
    if not org_hits:
        return 0.0

    quality_sum = 0.0
    total_docs = max(1, int(total_docs or 1))
    denom = float(np.log(total_docs + 1.0))
    if denom <= 0:
        denom = 1.0

    title_role_re = r"(?:高级|资深|首席|招聘|HR|人事|经理|总监|负责人|创始人|CEO|CTO|COO|CFO|VP)"
    for name in org_hits:
        kind = classify_org_candidate(name)
        if kind == "generic":
            continue

        quality = 1.0 if kind == "explicit" else 0.75
        if re.search(rf"{re.escape(name)}\s*[·•]\s*{title_role_re}", text, flags=re.IGNORECASE):
            quality += 0.20

        df = 1
        if candidate_df is not None:
            df = max(1, int(candidate_df.get(name, 1)))

        idf = float(np.log((total_docs + 1.0) / (df + 1.0)) / denom)
        idf = max(0.15, min(1.0, idf))
        quality_sum += quality * idf

    if quality_sum <= 0:
        return 0.0

    return min(0.32, 0.20 * quality_sum)


def should_score_body_term(term: str) -> bool:
    t = (term or "").strip().lower()
    if not t:
        return False

    if is_over_generic_term(t):
        return False
    if is_date_like_term(t):
        return False
    if is_sentence_like_term(t):
        return False

    # 中文词太短，默认不参与正文打分
    if re.fullmatch(r"[\u4e00-\u9fa5]+", t) and len(t) <= 2:
        return False

    # 纯英文/数字太短，也不参与正文打分
    if re.fullmatch(r"[a-z0-9_]+", t) and len(t) <= 2:
        return False

    return True
