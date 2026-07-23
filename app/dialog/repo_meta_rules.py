from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ai.capability_common import CATEGORY_COUNT_KEYWORDS, CATEGORY_KEYWORDS, normalize_meta_question
from ai.repo_meta.classifier_predicates import is_file_result_topic, looks_like_time_request

if TYPE_CHECKING:
    from app.dialog.state_machine import ConversationState


def is_structured_output_request(question: str) -> bool:
    q = (question or "").strip()

    patterns = [
        "时间线", "按时间顺序", "梳理一下", "整理一下",
        "列个清单", "分点总结", "做个表", "列出来",
        "给我个脉络", "帮我归纳一下", "详细看下", "详细说说",
    ]

    return any(p in q for p in patterns)


def is_system_capability_request(question: str) -> bool:
    q = (question or "").strip()
    patterns = [
        "你是谁", "介绍一下",
        "能干啥", "你能做什么", "能做什么", "可以做什么",
        "你可以做什么", "你的功能", "有什么功能", "有啥功能", "怎么用",
        "你能做啥", "能做啥", "做啥", "干啥",
    ]
    return any(p in q for p in patterns)


def is_repo_meta_request(question: str) -> bool:
    q = normalize_meta_question(question)
    if _looks_like_doc_inventory_listing_request(q):
        return True

    if looks_like_time_request(q):
        return True

    if _is_file_locator_query(q):
        return False

    has_doc_word = any(x in q for x in ["文件", "文档", "资料"])
    has_list_intent = any(x in q for x in ["列出", "列下", "列一下", "列出来", "罗列", "展开一下", "展开列一下"])
    has_topic_overview_intent = any(
        x in q
        for x in [
            "关于什么",
            "什么内容",
            "内容是什么",
            "什么主题",
            "主题是什么",
            "主要讲什么",
            "主要是什么",
            "主要是啥",
            "讲什么",
        ]
    )

    patterns = [
        "多少文件", "多少个文件", "文件数量",
        "多少文档", "多少个文档", "文档数量",
        "有多少文件", "有多少文档",
        "目前有多少文件", "目前有多少文档",
        "现在有多少文件", "现在有多少文档",

        "有哪些文件", "都有哪些文件", "文件清单",
        "有哪些文档", "都有哪些文档", "文档清单",

        "哪些类文档", "哪些类文件", "文档有哪些类", "文件有哪些类",
        "怎么分类", "如何分类", "分成哪些",
        "哪些格式", "都是什么格式", "分别是什么格式", "文件格式", "文档格式",

        "占多大空间", "总共多大", "总大小", "总体积", "占用空间", "总容量",

        "多少类文档", "多少类文件",
        "有多少类文档", "有多少类文件",
        "目前有多少类文档", "目前有多少类文件",
        "现在有多少类文档", "现在有多少类文件",
        "文档有多少类", "文件有多少类",
        "文档分几类", "文件分几类",
    ]
    if any(p in q for p in patterns):
        return True

    if has_doc_word and _looks_like_doc_inventory_listing_request(q):
        return True

    if has_doc_word and any(x in q for x in CATEGORY_KEYWORDS + CATEGORY_COUNT_KEYWORDS):
        return True

    if has_doc_word and has_topic_overview_intent:
        return True

    # 兜底：允许“列一下/列出来”触发 repo_meta，但必须显式提到文件/文档。
    if has_doc_word and has_list_intent:
        return True

    return False


def _looks_like_doc_inventory_listing_request(q: str) -> bool:
    normalized = (q or "").strip()
    if not normalized:
        return False

    patterns = (
        r"(?:当前|目前|现在).{0,4}存(?:的是?|是|有)?(?:哪些|什么)(?:文件|文档|资料)",
        r"^(?:当前|目前|现在)(?:有哪|有哪些|都有哪些)(?:文件|文档|资料)[？?]?$",
        r"^(?:有哪|有哪些|都有哪些)(?:文件|文档|资料)[？?]?$",
        r"^(?:文件|文档|资料)(?:有哪|有哪些)[？?]?$",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def _is_file_locator_query(q: str) -> bool:
    merged = re.sub(r"\s+", "", (q or "").lower())
    if not merged:
        return False

    if _looks_like_doc_inventory_listing_request(merged):
        return False

    direct_patterns = [
        "在哪个文件", "在那个文件", "是哪个文件", "是那个文件",
        "哪个文件", "哪些文件", "哪份文件", "文件里", "文件中",
        "在哪个文档", "在那个文档", "是哪个文档", "是那个文档",
        "哪个文档", "哪些文档",
        "在哪个记录", "是哪个记录", "哪个记录", "哪些记录",
    ]
    return any(p in merged for p in direct_patterns)


_CONTENT_LOOKUP_TARGET_PATTERNS = (
    r"(?:有哪些|有哪(?:些|个|几|位|家|条|项|种)|哪些|哪几个|哪几家|哪几位|哪几条|哪几项|哪个|哪位)(.+)$",
    r"(.+?)(?:有)?(?:哪些|哪几个|哪几家|哪几位|哪几条|哪几项)$",
)

_BARE_LOOKUP_TARGETS = {
    "", "还", "还有", "都", "分别", "具体", "其他", "其它", "别", "别的", "更多",
    "这", "那", "这个", "那个", "这些", "那些", "其中", "里面", "这里面",
}


def _is_repo_meta_lookup_target(target: str) -> bool:
    cleaned = re.sub(r"^(?:是|为|叫|属于)", "", target)
    cleaned = re.sub(r"(?:呢|吗|啊|呀|吧)$", "", cleaned)
    if cleaned in _BARE_LOOKUP_TARGETS:
        return True

    return bool(
        re.fullmatch(
            r"(?:类|类别|分类|大类|板块|方面|方向|主题|内容)"
            r"(?:最多|最少|数量|有多少|各有多少)?",
            cleaned,
        )
    )


def extract_content_lookup_target(question: str) -> str:
    """从内容枚举问句中提取开放目标；仓库元数据目标返回空串。"""
    q = normalize_meta_question(question)
    if not q or is_repo_meta_request(question):
        return ""

    for pattern in _CONTENT_LOOKUP_TARGET_PATTERNS:
        match = re.search(pattern, q)
        if not match:
            continue
        target = match.group(1).strip()
        if target and not _is_repo_meta_lookup_target(target):
            return re.sub(r"(?:呢|吗|啊|呀|吧)$", "", target)

    if (
        q.endswith("谁")
        and any(marker in q for marker in ("涉及", "包含", "包括", "提到", "提及", "记录", "出现"))
    ):
        return "谁"
    return ""


def is_content_lookup_request(question: str) -> bool:
    """识别带开放目标槽位的内容枚举，避免继承成仓库元数据追问。"""
    return bool(extract_content_lookup_target(question))


def is_entity_lookup_request(question: str) -> bool:
    """兼容旧调用名；实体目标由问句结构提取，不依赖业务词表。"""
    return is_content_lookup_request(question)


def _has_explicit_date_reference(text: str) -> bool:
    q = normalize_meta_question(text)
    if not q:
        return False

    date_patterns = (
        r"(?<!\d)(?:19|20)\d{2}[年./\-]\d{1,2}[月./\-]\d{1,2}(?:日|号)?",
        r"(?<!\d)\d{1,2}[月./\-]\d{1,2}(?:日|号)?",
        r"(?<!\d)\d{1,2}(?:日|号)(?!\d)",
    )
    return any(re.search(p, q) for p in date_patterns)


def _has_explicit_file_ref(text: str) -> bool:
    q = (text or "").strip()
    if not q:
        return False
    return bool(
        re.search(
            r"[A-Za-z0-9_\-\u4e00-\u9fa5\s]+?\.(?:txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx|png|jpg|jpeg|bmp|webp)",
            q,
            flags=re.IGNORECASE,
        )
    )


def _has_file_context_signal(text: str) -> bool:
    if _has_explicit_file_ref(text):
        return True
    q = normalize_meta_question(text)
    return any(x in q for x in ["文件", "文档", "资料", "简历", "合同", "报告", "清单"])


def looks_like_repo_size_consistency_followup(question: str, prev_question: str | None = None) -> bool:
    q = normalize_meta_question(question)
    has_size_word = any(x in q for x in ["大小", "体积", "容量", "占用", "字节", "kb", "mb", "gb"])
    has_consistency_word = any(x in q for x in ["一致", "一样", "相同", "同吗"])

    if not (has_size_word and has_consistency_word):
        return False

    if _has_file_context_signal(question):
        return True

    # 短追问：依赖上一轮文件上下文
    if len(q) <= 12 and _has_file_context_signal(prev_question):
        return True

    return False


def looks_like_repo_time_question(question: str, state: "ConversationState | None" = None) -> bool:
    q = normalize_meta_question(question)
    has_repo_meta_file_context = bool(
        state is not None
        and state.last_route == "repo_meta"
        and state.last_content_route == "repo_meta"
        and is_file_result_topic(state.last_local_topic)
        and state.last_answer_type == "enumeration_file"
        and state.last_result_set_entity_type == "文件"
        and state.last_result_set_items
        and state.last_result_set_selectable is True
    )
    return looks_like_time_request(
        q,
        has_repo_meta_file_context=has_repo_meta_file_context,
    )


def looks_like_repo_topic_question(question: str, state: "ConversationState | None" = None) -> bool:
    q = normalize_meta_question(question)
    if not q:
        return False

    topic_markers = (
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
    has_topic_intent = any(x in q for x in topic_markers)
    if not has_topic_intent:
        return False

    mentions_doc = any(x in q for x in ("文件", "文档", "资料"))
    if mentions_doc:
        return True

    if (
        state is not None
        and state.last_route == "repo_meta"
        and len(q) <= 18
    ):
        return True

    return False


def is_list_format_modifier(question: str) -> bool:
    """识别对上一轮列表的格式修饰请求，如"带时间""加上大小"等"""
    q = (question or "").strip()
    if len(q) > 15:
        return False

    patterns = [
        "带时间", "加时间", "加上时间", "要时间", "显示时间",
        "带大小", "加大小", "加上大小", "要大小", "显示大小",
        "带日期", "加日期", "加上日期",
        "按时间排", "按时间排序", "按日期排", "按日期排序",
        "按大小排", "按大小排序",
        "详细看下", "详细一点", "详细说说", "展开看下", "展开看看",
    ]
    return any(p in q for p in patterns)
