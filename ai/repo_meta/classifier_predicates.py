from __future__ import annotations

import re

from ai.capability_common import clean_text, contains_any, normalize_meta_question

TIME_LIST_INTENT_KEYWORDS = ("有哪些", "有哪", "哪些", "哪几个", "哪几份", "列出", "列一下", "列出来")
EMPTY_TOPIC_WORDS = {"文件", "文档", "资料", "内容"}

SIZE_CONSISTENCY_KEYWORDS = (
    "大小一致", "大小一样", "大小相同",
    "体积一致", "体积一样", "体积相同",
    "容量一致", "容量一样", "容量相同",
)

TOPIC_OVERVIEW_KEYWORDS = (
    "关于什么",
    "什么内容",
    "内容是什么",
    "什么主题",
    "主题是什么",
    "主要讲什么",
    "主要是什么",
    "主要是啥",
    "讲什么",
)

DEEPER_SUMMARY_KEYWORDS = (
    "还能再概括",
    "再概括",
    "更概括",
    "再总结",
    "再归纳",
    "再抽象",
    "一句话概括",
    "一句话总结",
    "整体上",
    "总体上",
    "本质上",
    "归根结底",
    "再上一层",
    "再往上",
)

TOPIC_META_NOISE_KEYWORDS = (
    "最近", "最新", "最早", "最晚", "最旧",
    "时间", "日期", "更新", "修改", "创建",
    "文件", "文档", "资料", "内容",
    "有哪些", "有哪", "哪些", "哪几个", "哪几份", "几个", "几份",
    "列出", "列一下", "列出来", "清单",
    "格式", "分类", "类别", "数量", "多少",
    "总共", "总体", "大小", "体积", "容量", "占用", "空间",
)


def _extract_topic_core(topic: str) -> str:
    t = normalize_meta_question(clean_text(topic))
    if not t:
        return ""

    for token in sorted(TOPIC_META_NOISE_KEYWORDS, key=len, reverse=True):
        t = t.replace(token, "")

    t = re.sub(r"[一二两三四五六七八九十\d个份条些几多]+", "", t)
    return t.strip()


def is_semantic_topic_candidate(topic: str) -> bool:
    core = _extract_topic_core(topic)
    if not core:
        return False
    if core in EMPTY_TOPIC_WORDS:
        return False
    return len(core) >= 2


def _has_explicit_date_reference(text: str) -> bool:
    q = normalize_meta_question(clean_text(text))
    if not q:
        return False

    patterns = (
        r"(?<!\d)(?:19|20)\d{2}[年/\-]\d{1,2}[月/\-]\d{1,2}(?:日|号)?",
        r"(?<!\d)\d{1,2}[月/\-]\d{1,2}(?:日|号)?",
        r"(?<!\d)\d{1,2}(?:日|号)(?!\d)",
    )
    return any(re.search(p, q) for p in patterns)


def looks_like_time_request(q: str, topic_candidate_valid: bool = False) -> bool:
    has_doc_word = any(x in q for x in ("文件", "文档", "资料"))
    has_format_word = ("格式" in q) or any(x in q for x in ("doc", "docx", "pdf", "txt", "md", "xls", "xlsx", "csv", "ppt", "pptx"))
    has_explicit_date = _has_explicit_date_reference(q)
    has_list_intent = any(x in q for x in TIME_LIST_INTENT_KEYWORDS + ("还有", "其他", "别的"))

    if has_explicit_date and has_doc_word and has_list_intent:
        return True

    has_time_word = any(
        x in q
        for x in (
            "最近", "最新", "最早", "最晚", "最旧",
            "最近更新", "最近修改", "更新时间",
            "修改时间", "创建时间", "时间", "日期",
        )
    )
    if not has_time_word:
        return False

    has_explicit_time_axis = any(x in q for x in ("时间", "日期", "更新", "修改", "创建"))
    if has_explicit_time_axis:
        return True

    if has_doc_word and has_time_word and not topic_candidate_valid:
        return True
    if has_time_word and has_format_word:
        return True
    if has_time_word and has_list_intent and len(q) <= 18 and not topic_candidate_valid:
        return True
    if has_time_word and len(q) <= 15 and any(x in q for x in ("它", "这", "那", "几")) and not topic_candidate_valid:
        return True
    return False


def _has_explicit_file_ref(text: str) -> bool:
    q = (text or "").strip()
    if not q:
        return False
    return bool(
        re.search(
            r"[A-Za-z0-9_\-\u4e00-\u9fa5\s]+?\.(?:txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx)",
            q,
            flags=re.IGNORECASE,
        )
    )


def is_size_consistency_request(question: str, last_user_question: str | None = None) -> bool:
    q = normalize_meta_question(clean_text(question))
    has_size_word = any(x in q for x in ("大小", "体积", "容量", "占用", "字节", "kb", "mb", "gb"))
    has_consistency_word = any(x in q for x in ("一致", "一样", "相同", "同吗"))

    if not (has_size_word and has_consistency_word):
        return False

    if contains_any(q, ("文件", "文档", "资料")) or _has_explicit_file_ref(question):
        return True

    if len(q) <= 12 and (
        contains_any(last_user_question, ("文件", "文档", "资料", "简历"))
        or _has_explicit_file_ref(last_user_question)
    ):
        return True

    return any(x in q for x in SIZE_CONSISTENCY_KEYWORDS)


def is_topic_overview_request(question: str, last_local_topic: str | None = None) -> bool:
    q = normalize_meta_question(clean_text(question))
    if not q:
        return False

    has_topic_intent = any(x in q for x in TOPIC_OVERVIEW_KEYWORDS)
    if not has_topic_intent:
        return False

    if any(x in q for x in ("文件", "文档", "资料")):
        return True

    if (
        last_local_topic in {
            "count",
            "format",
            "time",
            "total_size",
            "size_consistency",
            "list_files",
            "list_files_with_time",
            "list_files_by_topic",
            "category",
            "category_summary",
            "category_overview",
        }
        and len(q) <= 20
    ):
        return True

    return False


def is_deeper_category_summary_request(question: str, last_local_topic: str | None = None) -> bool:
    if last_local_topic not in {"category_summary", "category_overview"}:
        return False

    q = normalize_meta_question(clean_text(question))
    if not q:
        return False

    if contains_any(q, DEEPER_SUMMARY_KEYWORDS):
        return True

    if contains_any(q, ("概括", "总结", "归纳")) and any(x in q for x in ("再", "更", "还")) and len(q) <= 12:
        return True

    return False

