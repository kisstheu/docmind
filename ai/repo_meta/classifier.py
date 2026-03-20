from __future__ import annotations

import re

from ai.capability_common import (
    CATEGORY_CONFIRM_KEYWORDS,
    CATEGORY_KEYWORDS,
    CATEGORY_SUMMARY_KEYWORDS,
    LIST_FILE_KEYWORDS,
    TOTAL_SIZE_KEYWORDS,
    clean_text,
    contains_any, normalize_meta_question,
)

LIST_BY_TOPIC_PATTERNS = (
    r"列一下(.+?)的文件",
    r"列出(.+?)的文件",
    r"把(.+?)相关文件列出来",
    r"把(.+?)的文件列出来",
    r"(.+?)有哪些文件",
    r"(.+?)相关文档",
)

COUNT_KEYWORDS = ("多少文件", "多少个文件", "文件数量", "文档数量")
FORMAT_KEYWORDS = ("哪些格式", "文件格式", "文档格式", "支持格式")
TIME_KEYWORDS = (
    "最近更新",
    "最新文件", "最早文件",
    "最新文档", "最早文档",
    "最新的文件", "最早的文件",
    "最新的文档", "最早的文档",
    "最晚的文件", "最晚的文档",
)
LIST_FOLLOWUP_KEYWORDS = ("列一下", "列一下吧", "列出来", "展开一下", "展开列一下")
CATEGORY_FOLLOWUP_KEYWORDS = ("方面", "分类", "类别", "哪类", "怎么分", "如何分")
EMPTY_TOPIC_WORDS = {"文件", "文档", "资料", "内容"}


RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("count", COUNT_KEYWORDS),
    ("total_size", TOTAL_SIZE_KEYWORDS),
    ("format", FORMAT_KEYWORDS),
    ("time", TIME_KEYWORDS),
    ("list_files", LIST_FILE_KEYWORDS),
)


def is_category_summary_request(question: str) -> bool:
    return contains_any(question, CATEGORY_SUMMARY_KEYWORDS)



def is_category_confirmation_request(question: str) -> bool:
    return contains_any(question, CATEGORY_CONFIRM_KEYWORDS)



def is_followup_from_file_list(last_question: str | None, current_question: str) -> bool:
    return contains_any(last_question, LIST_FILE_KEYWORDS) and contains_any(current_question, CATEGORY_FOLLOWUP_KEYWORDS)



def is_followup_from_category(last_question: str | None, current_question: str) -> bool:
    category_context_keywords = CATEGORY_KEYWORDS + CATEGORY_SUMMARY_KEYWORDS
    return contains_any(last_question, category_context_keywords) and (
        is_category_summary_request(current_question) or is_category_confirmation_request(current_question)
    )



def is_followup_to_list_files(last_topic: str | None, current_question: str) -> bool:
    return last_topic in {"count", "list_files"} and contains_any(current_question, LIST_FOLLOWUP_KEYWORDS)



def extract_topic_from_list_request(question: str) -> str:
    q = (question or "").strip()
    for pattern in LIST_BY_TOPIC_PATTERNS:
        match = re.search(pattern, q)
        if not match:
            continue

        topic = match.group(1).strip()
        if topic and topic not in EMPTY_TOPIC_WORDS:
            return topic

    return ""



def classify_repo_meta_question(
    question: str,
    last_user_question: str | None = None,
    last_local_topic: str | None = None,
) -> str | None:
    q = normalize_meta_question(clean_text(question))

    topic_candidate = extract_topic_from_list_request(q)
    if topic_candidate:
        return "list_files_by_topic"

    for topic, keywords in RULES:
        if contains_any(q, keywords):
            return topic

    if is_followup_to_list_files(last_local_topic, q):
        return "list_files"

    if is_category_summary_request(q):
        return "category_summary"

    if is_category_confirmation_request(q):
        return "category_confirm"

    if contains_any(q, CATEGORY_KEYWORDS):
        return "category"

    if is_followup_from_file_list(last_user_question, q):
        return "category"

    if is_followup_from_category(last_user_question, q):
        if is_category_summary_request(q):
            return "category_summary"
        if is_category_confirmation_request(q):
            return "category_confirm"

    return None
