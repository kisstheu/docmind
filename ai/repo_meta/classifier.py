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

LIST_FORMAT_MODIFIERS = (
    "带时间", "加时间", "加上时间", "要时间", "显示时间",
    "带日期", "加日期", "加上日期",
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
    "文件最新", "文件最早", "文件最晚", "文件最旧",
    "文档最新", "文档最早", "文档最晚", "文档最旧",
)

SIZE_CONSISTENCY_KEYWORDS = (
    "大小一致", "大小一样", "大小相同",
    "体积一致", "体积一样", "体积相同",
    "容量一致", "容量一样", "容量相同",
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

LIST_FOLLOWUP_KEYWORDS = ("列一下", "列一下吧", "列出来", "展开一下", "展开列一下")
CATEGORY_FOLLOWUP_KEYWORDS = ("方面", "分类", "类别", "哪类", "怎么分", "如何分")
EMPTY_TOPIC_WORDS = {"文件", "文档", "资料", "内容"}
LIST_INTENT_KEYWORDS = ("列出", "列一下", "列下", "列出来", "清单", "罗列", "展开")
TIME_LIST_INTENT_KEYWORDS = ("有哪些", "有哪", "哪些", "哪几个", "哪几份", "列出", "列一下", "列出来")
TOPIC_META_NOISE_KEYWORDS = (
    "最近", "最新", "最早", "最晚", "最旧",
    "时间", "日期", "更新", "修改", "创建",
    "文件", "文档", "资料", "内容",
    "有哪些", "有哪", "哪些", "哪几个", "哪几份", "几个", "几份",
    "列出", "列一下", "列出来", "清单",
    "格式", "分类", "类别", "数量", "多少",
    "总共", "总体", "大小", "体积", "容量", "占用", "空间",
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
        if topic and topic not in EMPTY_TOPIC_WORDS:
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


def looks_like_time_request(q: str, topic_candidate_valid: bool = False) -> bool:
    has_doc_word = any(x in q for x in ("文件", "文档", "资料"))
    has_format_word = ("格式" in q) or any(x in q for x in (
        "doc", "docx", "pdf", "txt", "md",
        "xls", "xlsx", "csv",
        "ppt", "pptx",
    ))
    has_time_word = any(x in q for x in (
        "最近", "最新", "最早", "最晚", "最旧",
        "最近更新", "最近修改", "更新时间",
        "修改时间", "创建时间", "时间", "日期",
    ))
    if not has_time_word:
        return False

    has_list_intent = any(x in q for x in TIME_LIST_INTENT_KEYWORDS)
    has_explicit_time_axis = any(x in q for x in ("时间", "日期", "更新", "修改", "创建"))

    if has_explicit_time_axis:
        return True

    # “最近有哪些文件”这类短句，若没有可靠语义主题，优先按时间类处理。
    if has_doc_word and has_time_word and not topic_candidate_valid:
        return True

    if has_time_word and has_format_word:
        return True

    if has_time_word and has_list_intent and len(q) <= 18 and not topic_candidate_valid:
        return True

    if has_time_word and len(q) <= 15 and any(x in q for x in ("份", "个", "两", "三", "几")) and not topic_candidate_valid:
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

    # 短追问：依赖上一轮文件上下文
    if len(q) <= 12 and (
        contains_any(last_user_question, ("文件", "文档", "资料", "简历"))
        or _has_explicit_file_ref(last_user_question)
    ):
        return True

    return any(x in q for x in SIZE_CONSISTENCY_KEYWORDS)


def classify_repo_meta_question(
    question: str,
    last_user_question: str | None = None,
    last_local_topic: str | None = None,
) -> str | None:
    q = normalize_meta_question(clean_text(question))

    # “时间线/过程梳理”属于内容组织类请求，不应落到文件元信息列表。
    if contains_any(q, TIMELINE_REQUEST_KEYWORDS):
        print(f"[repo_meta分类] q={q} -> None(timeline_structured)")
        return None

    if is_name_content_mismatch_request(q):
        print(f"[repo_meta分类] q={q} -> name_content_mismatch")
        return "name_content_mismatch"

    # 对上一轮 list_files 的时间格式修饰
    if last_local_topic == "list_files" and any(x in q for x in LIST_FORMAT_MODIFIERS):
        print(f"[repo_meta分类] q={q} -> list_files_with_time")
        return "list_files_with_time"

    topic_candidate = extract_topic_from_list_request(q)
    topic_candidate_valid = is_semantic_topic_candidate(topic_candidate) if topic_candidate else False

    if is_size_consistency_request(question, last_user_question=last_user_question):
        print(f"[repo_meta分类] q={q} -> size_consistency")
        return "size_consistency"

    # 时间查询优先级高于主题列举，避免“最近时间有哪些文件”误判成 topic。
    if looks_like_time_request(q, topic_candidate_valid=topic_candidate_valid):
        print(f"[repo_meta分类] q={q} -> time")
        return "time"

    if topic_candidate_valid:
        topic = "list_files_by_topic"
        print(f"[repo_meta分类] q={q} -> {topic}")
        return topic

    if is_list_files_request(q):
        print(f"[repo_meta分类] q={q} -> list_files")
        return "list_files"

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
