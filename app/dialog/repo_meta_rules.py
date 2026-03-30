from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ai.capability_common import normalize_meta_question

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
    has_doc_word = any(x in q for x in ["文件", "文档", "资料"])
    has_list_intent = any(x in q for x in ["列出", "列下", "列一下", "列出来", "罗列", "展开一下", "展开列一下"])

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
        "哪些格式", "文件格式", "文档格式",

        "最近更新", "最新文件", "最早文件",
        "最新文档", "最早文档",
        "最近有哪些文件", "最近有哪些文档",
        "最近的有哪些文件", "最近的有哪些文档",
        "最近时间有哪些", "最近时间有哪些文件", "最近时间有哪些文档",
        "最新有哪些文件", "最早有哪些文件", "最晚有哪些文件",

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

    # 兜底：允许“列一下/列出来”触发 repo_meta，但必须显式提到文件/文档。
    if has_doc_word and has_list_intent:
        return True

    # 日期 + 文件列表意图，优先按仓库元信息处理。
    if has_doc_word and _has_explicit_date_reference(question) and any(
        x in q for x in ["有哪些", "有哪", "哪些", "哪几个", "哪几份", "还有", "其他", "别的"]
    ):
        return True

    return False


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

    # 显式提到文件/文档 + 时间信号
    mentions_doc = any(x in q for x in ["文件", "文档", "资料", "pdf", "txt", "docx"])
    has_explicit_date = _has_explicit_date_reference(question)
    time_signals = [
        "最近", "最新", "最早", "最晚", "最旧",
        "最近更新", "最近修改", "更新时间",
        "修改时间", "创建时间", "时间", "日期",
        "时间最新", "时间最早",
    ]
    has_time_signal = any(x in q for x in time_signals)
    has_list_intent = any(
        x in q for x in ["有哪些", "有哪", "哪些", "哪几个", "哪几份", "列出", "列一下", "列出来", "还有", "其他", "别的"]
    )
    has_explicit_time_axis = any(x in q for x in ["时间", "日期", "更新", "修改", "创建"])

    if has_time_signal and mentions_doc:
        return True

    if has_explicit_time_axis and has_time_signal:
        return True

    # 显式日期 + 文件范围 + 列表意图：按时间检索路由，避免误走内容追问。
    if has_explicit_date and mentions_doc and has_list_intent:
        return True

    # 上一轮在 repo_meta 上下文里，短问“最近的有哪些/最近时间有哪些”也按时间类处理
    if (
        state is not None
        and state.last_route == "repo_meta"
        and has_time_signal
        and has_list_intent
        and len(q) <= 18
    ):
        return True

    # 短句 + 时间信号 + 量词模式 = 独立的文件时间查询
    if len(q) <= 15:
        has_quantity = any(x in q for x in ["份", "个", "两", "三", "几"])
        if has_time_signal and has_quantity:
            return True

    # repo_meta 追问里，显式日期 + 列表意图也视为时间查询。
    if (
        state is not None
        and state.last_route == "repo_meta"
        and has_explicit_date
        and has_list_intent
        and len(q) <= 24
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
    ]
    return any(p in q for p in patterns)
