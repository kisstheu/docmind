from __future__ import annotations

import re

from ai.capability_common import (
    CATEGORY_BREAKDOWN_COUNT_KEYWORDS,
    CATEGORY_CONFIRM_KEYWORDS,
    CATEGORY_COUNT_KEYWORDS,
    CATEGORY_KEYWORDS,
    CATEGORY_SUMMARY_KEYWORDS,
    LIST_FILE_KEYWORDS,
    TOTAL_SIZE_KEYWORDS,
    clean_text,
    contains_any,
    normalize_meta_question,
)
from ai.repo_meta.classifier_predicates import (
    is_deeper_category_summary_request,
    is_semantic_topic_candidate,
    is_size_consistency_request,
    is_topic_overview_request,
    looks_like_time_request,
)

LIST_FORMAT_MODIFIERS = (
    "带时间", "加时间", "加上时间", "要时间", "显示时间",
    "带日期", "加日期", "加上日期",
)

LIST_BY_TOPIC_PATTERNS = (
    r"列一个(.+?)的文件",
    r"列出(.+?)的文件",
    r"列出(.+?)文件名",
    r"把(.+?)相关文件列出来",
    r"把(.+?)的文件列出来",
    r"把(.+?)文件名列出来",
    r"把(.+?)相关的文件名列出来",
    r"(.+?)有哪些文件",
    r"(.+?)有哪些文件名",
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
    "最近",
    "最近更新",
    "最近修改",
    "更新时间",
    "修改时间",
    "创建时间",
    "时间",
    "日期",
    "最新文件", "最早文件", "最晚文件", "最旧文件",
    "最新文档", "最早文档", "最晚文档", "最旧文档",
    "最新的文件", "最早的文件", "最晚的文件", "最旧的文件",
    "最新的文档", "最早的文档", "最晚的文档", "最旧的文档",
)

TIMELINE_REQUEST_KEYWORDS = (
    "时间线",
    "按时间顺序",
    "时间顺序",
    "梳理一下",
    "整理一下",
    "过程",
    "经过",
    "脉络",
)

LIST_FOLLOWUP_KEYWORDS = ("列一个", "列一下吧", "列出来", "展开一下", "展开列一个")
CATEGORY_FOLLOWUP_KEYWORDS = ("方面", "分类", "类别", "哪类", "怎么分", "如何分")
LIST_INTENT_KEYWORDS = ("列出", "列一个", "列下", "列出来", "清单", "罗列", "展开")
CATEGORY_DRILLDOWN_KEYWORDS = (
    "再拆分一下分类",
    "拆分一下分类",
    "再拆分一下",
    "继续拆分",
    "往下拆分",
    "再细分一下",
    "细分一下",
    "子分类",
    "再分一下类",
    "再拆一下",
)

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
        if topic and topic not in {"文件", "文档", "资料", "内容"}:
            return topic
    return ""


def is_name_content_mismatch_request(question: str) -> bool:
    q = normalize_meta_question(clean_text(question))
    has_name_word = any(x in q for x in ("文件名", "标题", "题目", "名称", "名字"))
    has_content_word = any(x in q for x in ("内容", "正文"))
    has_mismatch_word = any(x in q for x in ("不符", "不一致", "不匹配", "对不上", "冲突", "矛盾"))
    return has_name_word and has_content_word and has_mismatch_word


def is_list_files_request(question: str) -> bool:
    q = normalize_meta_question(clean_text(question))
    has_doc_word = any(x in q for x in ("文件", "文档", "资料"))
    has_list_intent = any(x in q for x in LIST_INTENT_KEYWORDS)
    return has_doc_word and has_list_intent


def is_category_drilldown_request(question: str, last_local_topic: str | None = None) -> bool:
    q = normalize_meta_question(clean_text(question))
    if not q:
        return False

    has_drilldown_intent = contains_any(q, CATEGORY_DRILLDOWN_KEYWORDS) or (
        "分类" in q and any(x in q for x in ("拆分", "细分", "往下", "展开"))
    )
    if not has_drilldown_intent:
        return False

    if last_local_topic in {"count", "list_files_by_topic", "category_count_breakdown"}:
        return True

    return last_local_topic in {"category_summary", "category_overview"} and any(
        x in q for x in ("这个板块", "这个分类", "这块", "这一类", "这里面")
    )


def classify_repo_meta_question(
    question: str,
    last_user_question: str | None = None,
    last_local_topic: str | None = None,
) -> str | None:
    q = normalize_meta_question(clean_text(question))

    if contains_any(q, TIMELINE_REQUEST_KEYWORDS):
        print(f"[repo_meta分类] q={q} -> None(timeline_structured)")
        return None

    if is_name_content_mismatch_request(q):
        print(f"[repo_meta分类] q={q} -> name_content_mismatch")
        return "name_content_mismatch"

    if last_local_topic == "list_files" and any(x in q for x in LIST_FORMAT_MODIFIERS):
        print(f"[repo_meta分类] q={q} -> list_files_with_time")
        return "list_files_with_time"

    if is_deeper_category_summary_request(question, last_local_topic=last_local_topic):
        print(f"[repo_meta分类] q={q} -> category_overview")
        return "category_overview"

    if is_category_drilldown_request(question, last_local_topic=last_local_topic):
        print(f"[repo_meta分类] q={q} -> category_drilldown")
        return "category_drilldown"

    topic_candidate = extract_topic_from_list_request(q)
    topic_candidate_valid = is_semantic_topic_candidate(topic_candidate) if topic_candidate else False

    if is_size_consistency_request(question, last_user_question=last_user_question):
        print(f"[repo_meta分类] q={q} -> size_consistency")
        return "size_consistency"

    if looks_like_time_request(q, topic_candidate_valid=topic_candidate_valid):
        print(f"[repo_meta分类] q={q} -> time")
        return "time"

    if topic_candidate_valid:
        print(f"[repo_meta分类] q={q} -> list_files_by_topic")
        return "list_files_by_topic"

    if is_list_files_request(q):
        print(f"[repo_meta分类] q={q} -> list_files")
        return "list_files"

    if is_topic_overview_request(question, last_local_topic=last_local_topic):
        print(f"[repo_meta分类] q={q} -> category")
        return "category"

    if last_local_topic in {"category", "category_summary", "category_overview"} and contains_any(
        q,
        CATEGORY_BREAKDOWN_COUNT_KEYWORDS,
    ):
        print(f"[repo_meta分类] q={q} -> category_count_breakdown")
        return "category_count_breakdown"

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

    if last_local_topic in {"category", "category_summary", "category_overview"} and contains_any(
        q,
        ("仓库里呢", "本地存的", "本地的"),
    ):
        return "category_overview" if last_local_topic == "category_overview" else "category_summary"

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
