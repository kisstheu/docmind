from __future__ import annotations

import re

from app.context_anchor import is_context_dependent_question
from app.dialog.state_machine import ConversationState
from app.dialog_utils import is_followup_question, is_summary_followup_request
from app.chat_text.file_lookup import looks_like_file_set_content_question

CONTEXTLESS_FOLLOWUP_REPLY = (
    "这个问题缺少明确主语或上下文，我先不调用远程模型。"
    "请补充具体对象后再问，例如：请再概括一下 3 月 4 日会议纪要。"
)
_CONTEXTLESS_FOLLOWUP_MARKERS = (
    "关于什么",
    "再概括",
    "还能再概括",
    "更概括",
    "再总结",
    "再归纳",
    "一句话概括",
    "一句话总结",
    "展开一下",
    "详细一点",
    "再详细",
    "继续",
    "然后呢",
    "还有吗",
)
_SUBJECT_HINT_TERMS = (
    "文件",
    "文档",
    "资料",
    "记录",
    "笔记",
    "截图",
    "会议",
    "纪要",
    "公司",
    "企业",
    "项目",
    "人物",
    "人名",
    "代码",
    "仓库",
    "目录",
)
_NO_CONTEXT_ANSWER_MARKERS = (
    "没有检索到足够可靠的参考片段",
    "信息不足",
    "没有可检索文档",
    "没有形成可检索片段",
    "先不调用远程模型",
)

_ANALYTIC_RETRIEVAL_MARKERS = (
    "为什么",
    "怎么",
    "如何",
    "分析",
    "总结",
    "比较",
    "区别",
    "主要集中",
    "集中在哪",
    "哪些方向",
    "技术方向",
    "方向之间",
    "之间的关系",
    "关系和组合",
    "组合情况",
    "分布",
    "趋势",
    "结构",
    "模式",
    "共现",
    "搭配",
    "哪一类",
    "哪类",
    "需求最多",
    "最频繁",
    "出现频率",
    "高频",
    "数量",
    "占比",
    "按数量",
    "按占比",
    "简单说明",
    "技术栈",
    "经验要求",
    "切入口",
    "门槛",
)


def _normalize_for_guard(text: str) -> str:
    q = (text or "").strip().lower()
    return re.sub(r"[，。！？?.!?\s]+", "", q)


def _looks_like_contextless_followup_question(question: str) -> bool:
    q = _normalize_for_guard(question)
    if not q or len(q) > 24:
        return False
    if any(term in q for term in _SUBJECT_HINT_TERMS):
        return False
    if any(marker in q for marker in _CONTEXTLESS_FOLLOWUP_MARKERS):
        return True
    return is_followup_question(question)


def _has_usable_followup_context(state: ConversationState) -> bool:
    last_answer = (state.last_answer_text or state.last_answer_preview or "").strip()
    if last_answer and not any(marker in last_answer for marker in _NO_CONTEXT_ANSWER_MARKERS):
        return True

    last_query = (state.last_effective_search_query or "").strip()
    if last_query and not _looks_like_contextless_followup_question(last_query):
        return True

    last_content_q = (state.last_content_user_question or "").strip()
    if last_content_q and not _looks_like_contextless_followup_question(last_content_q):
        return True
    return False


def looks_like_analytic_retrieval_question(question: str) -> bool:
    q = _normalize_for_guard(question)
    if not q:
        return False
    if is_summary_followup_request(question):
        return True
    if looks_like_file_set_content_question(question):
        return True
    if re.search(
        r"(?:(?:\u54ea\u4e9b|\u54ea\u51e0(?:\u4e2a|\u4efd|\u5f20)?)(?:\u6587\u4ef6|\u6587\u6863|\u8bb0\u5f55|\u622a\u56fe)"
        r"|(?:\u54ea\u4e2a|\u54ea\u4efd|\u54ea\u7bc7|\u54ea\u5f20)(?:\u6587\u4ef6|\u6587\u6863|\u8bb0\u5f55|\u622a\u56fe)?"
        r"|\u5728\u54ea(?:\u4e2a|\u4efd|\u7bc7|\u5f20)?(?:\u6587\u4ef6|\u6587\u6863|\u8bb0\u5f55|\u622a\u56fe))",
        q,
    ):
        return False
    if any(marker in q for marker in _ANALYTIC_RETRIEVAL_MARKERS):
        return True
    if re.search(r"\u8fd9\u4e9b.*\u65b9\u5411", q):
        return True
    if re.search(r"\u54ea\u4e9b.*\u65b9\u5411", q):
        return True
    if re.search(r"\u54ea.*\u7c7b.*\u6700\u591a", q):
        return True
    if re.search(r"(?:\u8981\u6c42|\u6280\u80fd|\u638c\u63e1|\u6280\u672f\u6808|\u80fd\u529b|\u7ecf\u9a8c).*(?:\u54ea\u4e9b|\u4ec0\u4e48|\u76f8\u5bf9\u8f83\u4f4e|\u5207\u5165\u53e3)", q):
        return True
    if re.search(r"\u54ea.*(?:\u6280\u672f|\u6280\u80fd)", q):
        return True
    if re.search(r"\u66f4\u5bb9\u6613.*\u5207\u5165\u53e3", q):
        return True
    if re.search(r".*\u4e4b\u95f4.*\u5173\u7cfb", q):
        return True
    return False


def is_simple_retrieval_turn(question: str, event_name: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if looks_like_analytic_retrieval_question(q):
        return False
    if event_name in {"entity_lookup_followup", "result_set_followup", "result_set_expansion_followup"}:
        return True
    if event_name in {"content_followup", "action_request"} and len(q) <= 64:
        return True
    if event_name == "unknown" and len(q) <= 28:
        return True
    return False
