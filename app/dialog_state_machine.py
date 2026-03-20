from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.dialog_utils import (
    is_followup_question,
    is_repo_meta_confirmation,
    is_action_request,
    is_judgment_request,
    is_relationship_analysis_request,
    is_query_correction,
)
from app.context_anchor import is_context_dependent_question


# =========================
# 状态定义
# =========================

@dataclass
class ConversationState:
    mode: str = "idle"
    last_user_question: str | None = None
    last_route: str | None = None
    last_local_topic: str | None = None
    last_answer_preview: str | None = None

    last_content_user_question: str | None = None
    last_content_route: str | None = None
    last_content_topic: str | None = None

    last_effective_search_query: str | None = None


# =========================
# 事件定义
# =========================

@dataclass
class DialogEvent:
    name: str
    route_hint: Optional[str] = None
    merged_query: Optional[str] = None


# =========================
# 结构化请求识别
# =========================

def is_structured_output_request(question: str) -> bool:
    q = (question or "").strip()

    patterns = [
        "时间线", "按时间顺序", "梳理一下", "整理一下",
        "列个清单", "分点总结", "做个表", "列出来",
        "给我个脉络", "帮我归纳一下", "详细看下", "详细说说",
    ]

    return any(p in q for p in patterns)


# =========================
# 事件识别核心
# =========================

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
    q = (question or "").strip()

    patterns = [
        "多少文件", "多少个文件", "文件数量", "文档数量",
        "有哪些文件", "都有哪些文件", "文件清单",
        "有哪些文档", "都有哪些文档", "文档清单",
        "列一下", "列出来",
        "哪些类文档", "哪些类文件", "文档有哪些类", "文件有哪些类",
        "怎么分类", "如何分类", "分成哪些",
        "哪些格式", "文件格式", "文档格式",
        "最近更新", "最新文件", "最早文件",

        "占多大空间", "总共多大", "总大小", "总体积", "占用空间", "总容量",
    ]
    return any(p in q for p in patterns)

def detect_dialog_event(question: str, state: ConversationState) -> DialogEvent:
    prev_q = state.last_content_user_question
    prev_route = state.last_content_route
    last_topic = state.last_local_topic

    if is_system_capability_request(question):
        return DialogEvent(name="system_capability", route_hint="system_capability")
    if is_repo_meta_request(question):
        return DialogEvent(name="repo_meta_request", route_hint="repo_meta")
    # === 1. 纠偏
    if is_query_correction(question) and prev_q:
        return DialogEvent(
            name="query_correction",
            route_hint="normal_retrieval",
            merged_query=f"{prev_q} {question}",
        )

    # === 2. 关系判断
    if is_relationship_analysis_request(question):
        return DialogEvent(name="relationship_analysis", route_hint="normal_retrieval")

    # === 3. 结构化请求
    if is_structured_output_request(question):
        if prev_route == "normal_retrieval" and prev_q:
            return DialogEvent(
                name="structured_request",
                route_hint="normal_retrieval",
                merged_query=f"{prev_q} {question}",
            )
        return DialogEvent(name="structured_request", route_hint="normal_retrieval")

    # === 4. 动作请求
    if is_action_request(question):
        if prev_route == "normal_retrieval" and prev_q and len(question.strip()) <= 12:
            return DialogEvent(
                name="action_request",
                route_hint="normal_retrieval",
                merged_query=f"{prev_q} {question}",
            )
        return DialogEvent(name="action_request", route_hint="normal_retrieval")

    # === 5. 判断请求
    if is_judgment_request(question):
        if prev_route == "normal_retrieval" and prev_q:
            return DialogEvent(
                name="judgment_request",
                route_hint="normal_retrieval",
                merged_query=f"{prev_q} {question}",
            )
        return DialogEvent(name="judgment_request", route_hint="normal_retrieval")

    # === 6. repo_meta 继承
    if prev_route == "repo_meta":
        if is_followup_question(question):
            return DialogEvent(name="repo_followup", route_hint="repo_meta")

        if (
            last_topic in {"category", "category_summary", "category_confirm"}
            and is_repo_meta_confirmation(question)
        ):
            return DialogEvent(name="repo_confirm", route_hint="repo_meta")

        if (
            len(question.strip()) <= 12
            and any(x in question for x in ["粗", "细", "大类", "概括", "方面", "分类"])
        ):
            return DialogEvent(name="repo_followup", route_hint="repo_meta")

    # === 7. 内容追问继承
    if prev_route == "normal_retrieval" and is_context_dependent_question(question, state.last_effective_search_query):
        if prev_q:
            return DialogEvent(
                name="content_followup",
                route_hint="normal_retrieval",
                merged_query=f"{prev_q} {question}",
            )
        return DialogEvent(name="content_followup", route_hint="normal_retrieval")

    # === 8. 默认
    return DialogEvent(name="unknown", route_hint=None)


# =========================
# 状态转移（轻量）
# =========================

def apply_event_to_state(state: ConversationState, event: DialogEvent) -> ConversationState:
    new_state = ConversationState(
        mode=state.mode,
        last_route=state.last_route,
        last_local_topic=state.last_local_topic,
        last_content_user_question=state.last_content_user_question,
        last_content_route=state.last_content_route,
        last_content_topic=state.last_content_topic,
    )

    if event.route_hint == "repo_meta":
        new_state.mode = "repo_meta"
    elif event.route_hint == "normal_retrieval":
        new_state.mode = "content"
    elif event.name == "smalltalk":
        new_state.mode = "smalltalk"

    return new_state