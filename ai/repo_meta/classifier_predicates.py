from __future__ import annotations

import re

from ai.capability_common import clean_text, contains_any, normalize_meta_question

TIME_LIST_INTENT_KEYWORDS = ("有哪些", "有哪", "哪些", "哪几个", "哪几份", "列出", "列一下", "列出来")
EMPTY_TOPIC_WORDS = {"文件", "文档", "资料", "内容"}

FILE_RESULT_TOPICS = frozenset({
    "list_files",
    "list_files_with_time",
    "list_files_by_topic",
    "time",
})
FILE_TIME_SCOPE_KEYWORDS = ("文件", "文档", "资料", "笔记", "目录")
FILE_TIME_ACTION_KEYWORDS = ("创建", "修改", "更新", "新增", "删除", "重命名", "版本")
FILE_TIME_SORT_KEYWORDS = ("最近", "最新", "最早", "最晚", "最旧")
FILE_TIME_QUERY_KEYWORDS = ("时间", "日期", "什么时候", "何时", "哪天", "何日")

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


def _has_file_format_scope(text: str) -> bool:
    return bool(
        re.search(
            r"(?:^|[^a-z0-9])(?:docx?|pdf|txt|md|xlsx?|csv|pptx?)(?:$|[^a-z0-9])",
            text,
            flags=re.IGNORECASE,
        )
    )


def _has_named_month_range(text: str) -> bool:
    return bool(
        re.search(
            r"(?:[一二三四五六七八九十]{1,3}|[1-9]|1[0-2])月(?:份)?"
            r"(?:以后|之后|以来|之前|以前|起|开始)",
            text,
        )
    )


def _has_relative_date_reference(text: str) -> bool:
    return contains_any(
        text,
        (
            "今天", "昨天", "前天",
            "本周", "上周", "这周",
            "本月", "上月", "这个月", "上个月",
            "今年", "去年",
        ),
    )


def is_file_result_topic(topic: str | None) -> bool:
    return (topic or "").strip() in FILE_RESULT_TOPICS


def _looks_like_explicit_file_sort_request(text: str) -> bool:
    scope = r"(?:文件|文档|资料|笔记|目录|docx?|pdf|txt|md|xlsx?|csv|pptx?)"
    sort = r"(?:最近|最新|最早|最晚|最旧)"
    patterns = (
        rf"^(?:找出?|列出|查看|看看)?{sort}(?:的)?{scope}"
        rf"(?:是什么|有哪些|是哪(?:个|份)|吗|呢)?$",
        rf"^{sort}(?:的)?(?:时间|日期)?"
        rf"(?:有哪|有哪些|哪些|列出){scope}$",
        rf"^(?:哪个|哪份|哪些)?{scope}(?:中|里|里面)?"
        rf"(?:哪个|哪份|哪些)?(?:是|为)?{sort}(?:的)?(?:一个|一份)?$",
    )
    return any(re.fullmatch(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def looks_like_time_request(
    q: str,
    topic_candidate_valid: bool = False,
    has_repo_meta_file_context: bool = False,
) -> bool:
    normalized = normalize_meta_question(clean_text(q))
    if not normalized:
        return False

    has_file_scope = (
        contains_any(normalized, FILE_TIME_SCOPE_KEYWORDS)
        or _has_file_format_scope(normalized)
    )
    has_metadata_action = contains_any(normalized, FILE_TIME_ACTION_KEYWORDS)
    has_sort_signal = contains_any(normalized, FILE_TIME_SORT_KEYWORDS)
    has_time_query = contains_any(normalized, FILE_TIME_QUERY_KEYWORDS)
    has_date_range = (
        _has_explicit_date_reference(normalized)
        or _has_named_month_range(normalized)
        or _has_relative_date_reference(normalized)
    )
    has_list_intent = contains_any(
        normalized,
        TIME_LIST_INTENT_KEYWORDS + ("还有", "其他", "别的", "找出", "查找", "筛选"),
    )

    if has_repo_meta_file_context:
        contextual_followups = {
            "最近",
            "最近呢",
            "最近有哪些",
            "最近时间有哪些",
            "最新",
            "最新呢",
            "最早",
            "最早呢",
        }
        if normalized in contextual_followups:
            return True

    if has_file_scope and has_date_range and has_list_intent:
        return True

    if has_file_scope and has_metadata_action and (
        has_time_query or has_date_range or has_list_intent or has_sort_signal
    ):
        return True

    return (
        has_sort_signal
        and not topic_candidate_valid
        and _looks_like_explicit_file_sort_request(normalized)
    )


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
