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
    is_file_result_topic,
    is_semantic_topic_candidate,
    is_size_consistency_request,
    is_topic_overview_request,
    looks_like_time_request,
)

LIST_FORMAT_MODIFIERS = (
    "带时间", "加时间", "加上时间", "要时间", "显示时间",
    "带日期", "加日期", "加上日期",
)
LIST_DETAIL_MODIFIERS = (
    "详细看下", "详细一点", "详细些", "详细说说",
    "展开看下", "展开看看",
)

_FILE_LIST_TOPIC_META_TERMS = {"类型", "格式", "类别", "分类", "方面", "方向", "数量", "大小", "体积", "容量"}
_FILE_LIST_OBJECT_PATTERN = r"(?:文件名|文档名|资料名|文件|文档|资料)"
_FILE_LIST_QUESTION_INTENT_PATTERN = r"(?:有些什么|有哪一些|有哪些|有什么|有哪)"
_FILE_LIST_COMMAND_PATTERN = r"(?:列出来|列一下|列一个|列出|列下|罗列|清单)"
_FILE_LIST_POLITE_PREFIX_PATTERN = r"(?:(?:请|麻烦|帮我|请帮我|麻烦帮我))?"
_FILE_LIST_SLOT_PATTERNS = (
    rf"{_FILE_LIST_POLITE_PREFIX_PATTERN}(?:把)?"
    rf"{_FILE_LIST_COMMAND_PATTERN}(?P<slot>.*?){_FILE_LIST_OBJECT_PATTERN}",
    rf"{_FILE_LIST_POLITE_PREFIX_PATTERN}(?:把)?(?P<slot>.*?)"
    rf"{_FILE_LIST_OBJECT_PATTERN}{_FILE_LIST_COMMAND_PATTERN}",
    rf"(?P<slot>.*?){_FILE_LIST_QUESTION_INTENT_PATTERN}{_FILE_LIST_OBJECT_PATTERN}",
    rf"(?P<slot>.*?){_FILE_LIST_OBJECT_PATTERN}{_FILE_LIST_QUESTION_INTENT_PATTERN}",
    rf"(?P<slot>.+?)(?:相关){_FILE_LIST_OBJECT_PATTERN}",
)
_GENERIC_REPOSITORY_SCOPE = (
    r"(?:(?:当前|目前|现在))?(?:我(?:的)?)?(?:(?:整个|全部|所有))?"
    r"(?:知识库|库)(?:里|中|内)?(?:(?:一共|总共|全部|所有|都))*"
)
_GENERIC_STATUS_SCOPE = (
    r"(?:(?:当前|目前|现在)(?:(?:一共|总共|全部|所有|都))*|"
    r"(?:(?:一共|总共|全部|所有|都))+)"
)


def _normalize_file_list_question(question: str) -> str:
    return re.sub(r"[？?！!，,。.、；;：:\s]+", "", clean_text(question))


def _is_generic_repository_scope(slot: str) -> bool:
    value = (slot or "").strip()
    if not value:
        return True
    return bool(
        re.fullmatch(_GENERIC_REPOSITORY_SCOPE, value)
        or re.fullmatch(_GENERIC_STATUS_SCOPE, value)
    )


def _normalize_file_list_slot(slot: str) -> str:
    value = (slot or "").strip("的里中内上，。！？；：,.!?;: ")
    value = re.sub(r"^(?:当前|目前|现在)", "", value)
    value = re.sub(r"^(?:这个|这份|该)", "", value)
    value = re.sub(r"^(?:关于|有关)", "", value)
    value = re.sub(r"(?:相关|方面|有关)$", "", value)
    return value.strip("的里中内上，。！？；：,.!?;: ")


COUNT_KEYWORDS = (
    "多少文件", "多少个文件", "文件数量",
    "多少文档", "多少个文档", "文档数量",
    "有多少文件", "有多少文档",
    "目前有多少文件", "目前有多少文档",
    "现在有多少文件", "现在有多少文档",
    "总共有多少文件", "总共有多少文档",
)

FORMAT_KEYWORDS = (
    "哪些格式", "都是什么格式", "分别是什么格式", "文件格式", "文档格式", "支持格式",
    "doc", "docx", "pdf", "txt", "md",
    "xls", "xlsx", "csv",
    "ppt", "pptx",
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
    ("list_files", LIST_FILE_KEYWORDS),
)


def is_category_summary_request(question: str) -> bool:
    return contains_any(question, CATEGORY_SUMMARY_KEYWORDS)


def is_category_confirmation_request(question: str) -> bool:
    return contains_any(question, CATEGORY_CONFIRM_KEYWORDS)


def is_count_with_format_request(question: str) -> bool:
    q = normalize_meta_question(clean_text(question))
    has_global_scope = contains_any(
        q,
        ("知识库", "当前", "目前", "现在", "这些资料", "这些文档", "这些文件"),
    ) or q.startswith(("有多少", "多少", "总共有多少"))
    return has_global_scope and contains_any(q, COUNT_KEYWORDS) and ("格式" in q or contains_any(q, FORMAT_KEYWORDS))


def is_followup_from_file_list(last_question: str | None, current_question: str) -> bool:
    return contains_any(last_question, LIST_FILE_KEYWORDS) and contains_any(current_question, CATEGORY_FOLLOWUP_KEYWORDS)


def is_followup_from_category(last_question: str | None, current_question: str) -> bool:
    category_context_keywords = CATEGORY_KEYWORDS + CATEGORY_SUMMARY_KEYWORDS
    return contains_any(last_question, category_context_keywords) and (
        is_category_summary_request(current_question) or is_category_confirmation_request(current_question)
    )


def is_followup_to_list_files(last_topic: str | None, current_question: str) -> bool:
    return last_topic in {"count", "list_files"} and contains_any(current_question, LIST_FOLLOWUP_KEYWORDS)


def parse_file_list_request(question: str) -> str | None:
    """Return ``None`` for non-list, ``""`` for repo-wide, or the intact topic."""
    q = _normalize_file_list_question(question)
    if not q:
        return None

    for pattern in _FILE_LIST_SLOT_PATTERNS:
        match = re.fullmatch(pattern, q)
        if not match:
            continue
        raw_slot = match.group("slot")
        if _is_generic_repository_scope(raw_slot):
            return ""
        topic = _normalize_file_list_slot(raw_slot)
        if _is_generic_repository_scope(topic):
            return ""
        if topic and topic not in _FILE_LIST_TOPIC_META_TERMS:
            return topic
        return None

    scope_without_object = re.fullmatch(
        rf"(?P<slot>.+?){_FILE_LIST_QUESTION_INTENT_PATTERN}",
        q,
    )
    if scope_without_object and _is_generic_repository_scope(scope_without_object.group("slot")):
        return ""

    return None


def extract_topic_from_list_request(question: str) -> str:
    topic = parse_file_list_request(question)
    return topic or ""


def is_name_content_mismatch_request(question: str) -> bool:
    q = normalize_meta_question(clean_text(question))
    has_name_word = any(x in q for x in ("文件名", "标题", "题目", "名称", "名字"))
    has_content_word = any(x in q for x in ("内容", "正文"))
    has_mismatch_word = any(x in q for x in ("不符", "不一致", "不匹配", "对不上", "冲突", "矛盾"))
    return has_name_word and has_content_word and has_mismatch_word


def is_list_files_request(question: str) -> bool:
    if parse_file_list_request(question) == "":
        return True

    q = normalize_meta_question(clean_text(question))
    has_doc_word = any(x in q for x in ("文件", "文档", "资料"))
    has_list_intent = any(x in q for x in LIST_INTENT_KEYWORDS)
    return (has_doc_word and has_list_intent) or _looks_like_doc_inventory_listing_request(q)


def _looks_like_doc_inventory_listing_request(q: str) -> bool:
    normalized = (q or "").strip()
    if not normalized:
        return False

    patterns = (
        r"(?:当前|目前|现在).{0,4}存(?:的是?|是|有)?(?:哪些|什么)(?:文件|文档|资料)",
        r"^(?:有哪|有哪些|都有哪些)(?:文件|文档|资料)[？?]?$",
        r"^(?:文件|文档|资料)(?:有哪|有哪些)[？?]?$",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


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

    if last_local_topic == "list_files" and contains_any(q, LIST_DETAIL_MODIFIERS):
        print(f"[repo_meta分类] q={q} -> list_files_with_time(detail)")
        return "list_files_with_time"

    if is_deeper_category_summary_request(question, last_local_topic=last_local_topic):
        print(f"[repo_meta分类] q={q} -> category_overview")
        return "category_overview"

    if is_category_drilldown_request(question, last_local_topic=last_local_topic):
        print(f"[repo_meta分类] q={q} -> category_drilldown")
        return "category_drilldown"

    file_list_topic = parse_file_list_request(question)
    topic_candidate = file_list_topic or ""
    topic_candidate_valid = is_semantic_topic_candidate(topic_candidate) if topic_candidate else False

    if is_size_consistency_request(question, last_user_question=last_user_question):
        print(f"[repo_meta分类] q={q} -> size_consistency")
        return "size_consistency"

    if looks_like_time_request(
        q,
        topic_candidate_valid=topic_candidate_valid,
        has_repo_meta_file_context=is_file_result_topic(last_local_topic),
    ):
        print(f"[repo_meta分类] q={q} -> time")
        return "time"

    if file_list_topic == "":
        print(f"[repo_meta分类] q={q} -> list_files")
        return "list_files"

    if topic_candidate_valid:
        print(f"[repo_meta分类] q={q} -> list_files_by_topic")
        return "list_files_by_topic"

    if is_count_with_format_request(q):
        print(f"[repo_meta分类] q={q} -> count_with_format")
        return "count_with_format"

    if is_list_files_request(question):
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
