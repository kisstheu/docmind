from __future__ import annotations

import re

from ai.capability_common import (
    CATEGORY_CONFIRM_KEYWORDS,
    CATEGORY_KEYWORDS,
    CATEGORY_SUMMARY_KEYWORDS,
    LIST_FILE_KEYWORDS,
    TOTAL_SIZE_KEYWORDS,
    clean_text,
    contains_any,
    normalize_meta_question,
    CATEGORY_COUNT_KEYWORDS,
)

LIST_BY_TOPIC_PATTERNS = (
    r"列一下(.+?)的文件",
    r"列出(.+?)的文件",
    r"把(.+?)相关文件列出来",
    r"把(.+?)的文件列出来",
    r"(.+?)有哪些文件",
    r"(.+?)相关文档",
)

COUNT_KEYWORDS = (
    "多少文件", "多少个文件", "文件数量",
    "多少文档", "多少个文档", "文档数量",
    "有多少文件", "有多少文档",
    "目前有多少文件", "目前有多少文档",
    "现在有多少文件", "现在有多少文档",
    "总共有多少文件", "总共有多少文档",
)

FORMAT_KEYWORDS = (
    "哪些格式", "文件格式", "文档格式", "支持格式",
    "doc", "docx", "pdf", "txt", "md",
    "xls", "xlsx", "csv",
    "ppt", "pptx",
)

TIME_KEYWORDS = (
    "最近更新",
    "最近修改",
    "修改时间",
    "创建时间",
    "最新文件", "最早文件", "最晚文件",
    "最新文档", "最早文档", "最晚文档",
    "最新的文件", "最早的文件", "最晚的文件",
    "最新的文档", "最早的文档", "最晚的文档",
    "文件最新", "文件最早", "文件最晚",
    "文档最新", "文档最早", "文档最晚",
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
        topic = "list_files_by_topic"
        print(f"[repo_meta分类] q={q} -> {topic}")
        return topic

    has_doc_word = any(x in q for x in ("文件", "文档", "资料"))
    has_format_word = ("格式" in q) or any(x in q for x in (
        "doc", "docx", "pdf", "txt", "md",
        "xls", "xlsx", "csv",
        "ppt", "pptx",
    ))
    has_time_word = any(x in q for x in (
        "最新", "最早", "最晚",
        "最近更新", "最近修改",
        "修改时间", "创建时间",
    ))

    if (has_doc_word or has_format_word) and has_time_word:
        print(f"[repo_meta分类] q={q} -> time")
        return "time"

    for topic, keywords in RULES:
        if contains_any(q, keywords):
            print(f"[repo_meta分类] q={q} -> {topic}")
            return topic

    if is_followup_to_list_files(last_local_topic, q):
        print(f"[repo_meta分类] q={q} -> list_files")
        return "list_files"

    if is_category_summary_request(q):
        print(f"[repo_meta分类] q={q} -> category_summary")
        return "category_summary"

    if last_local_topic in {"category", "category_summary"} and contains_any(q, ("仓库里呢", "本地存的", "本地的")):
        return "category_summary"

    if is_category_confirmation_request(q):
        print(f"[repo_meta分类] q={q} -> category_confirm")
        return "category_confirm"

    if contains_any(q, CATEGORY_COUNT_KEYWORDS):
        return "category_summary"

    if contains_any(q, CATEGORY_KEYWORDS):
        print(f"[repo_meta分类] q={q} -> category")
        return "category"

    if is_followup_from_file_list(last_user_question, q):
        print(f"[repo_meta分类] q={q} -> category")
        return "category"

    if is_followup_from_category(last_user_question, q):
        if is_category_summary_request(q):
            print(f"[repo_meta分类] q={q} -> category_summary")
            return "category_summary"
        if is_category_confirmation_request(q):
            print(f"[repo_meta分类] q={q} -> category_confirm")
            return "category_confirm"

    print(f"[repo_meta分类] q={q} -> None")
    return None