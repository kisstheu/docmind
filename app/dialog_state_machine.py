from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

from ai.capability_common import normalize_meta_question
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
    last_answer_text: str | None = None
    last_answer_type: str | None = None
    last_result_set_query: str | None = None
    last_result_set_items: list[str] | None = None
    last_result_set_entity_type: str | None = None


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


RESULT_SET_FOLLOWUP_PATTERNS = [
    r"^哪些是",
    r"^哪个是",
    r"^哪几个是",
    r"^哪些属于",
    r"^哪些不是",
    r"^哪个不是",
    r"^其中哪些",
    r"^这里面哪些",
    r"^其中哪个",
    r"^这里面哪个",
    r"^哪些提到",
    r"^哪些涉及",
    r"^哪些和.*有关",
    r"^哪些与.*有关",
    r"^哪个最早",
    r"^哪个最新",
    r"^哪个最晚",
    r"^都有哪些",
    r"^有哪些",
    r"^分别有哪些",
    r"^分别是哪些",
    r"^具体有哪些",
    r"^具体是哪些",
    r"^都是什么",
    r"^分别是什么",
    r"^列一下",
    r"^展开说",
    r"^还有别的",
    r"^还有哪些",
    r"^哪几个",
    r"^哪几家",
    r"^目前知道的是哪几个",
    r"^目前知道的是哪几家",
    r"^所以.*哪几个",
    r"^所以.*哪几家",
]
RESULT_SET_CONTINUATION_PATTERNS = [
    r"^还有吗",
    r"^还有没有",
    r"^还有别的",
    r"^还有其他",
    r"^还有哪些",
    r"^还有别的吗",
    r"^还有其他的吗",
    r"^还包括",
    r"^除此之外",
    r"^另外还有",
]

def looks_like_result_set_followup(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False

    return (
        any(re.search(p, q) for p in RESULT_SET_FOLLOWUP_PATTERNS)
        or any(re.search(p, q) for p in RESULT_SET_CONTINUATION_PATTERNS)
    )


def last_turn_looks_like_enumeration(last_answer: str | None) -> bool:
    if not last_answer:
        return False

    text = last_answer.strip()

    numbered_lines = re.findall(r"(?:^|\n)\s*(?:\d+[.、]|[-*•])\s*", text)
    if len(numbered_lines) >= 2:
        return True

    if re.search(r"明确提到了\d+[家个项份条]", text):
        return True

    if re.search(r"提到了以下\d*[家个项份条]?", text):
        return True

    if re.search(r"以下\d+[家个项份条]", text):
        return True

    if "以下公司" in text or "提到了以下公司" in text:
        return True

    if "如下：" in text or "如下:" in text or "分别如下" in text:
        return True

    if text.count("有限公司") >= 2:
        return True

    return False
def infer_result_set_anchor(
    last_user_question: str | None,
    last_answer_type: str | None,
) -> str | None:
    prev = (last_user_question or "").strip()

    if last_answer_type == "enumeration_company":
        return "文档里提到的公司"
    if last_answer_type == "enumeration_file":
        return "上一轮提到的文件"
    if last_answer_type == "enumeration_person":
        return "上一轮提到的人物"

    if not prev:
        return None

    if "公司" in prev:
        return "文档里提到的公司"
    if "文件" in prev or "文档" in prev:
        return "上一轮提到的文件"
    if "人" in prev or "人物" in prev:
        return "上一轮提到的人物"

    return None


def build_result_set_followup_query(
    question: str,
    last_user_question: str | None,
    last_answer_type: str | None,
    last_result_set_items: list[str] | None = None,
    last_result_set_entity_type: str | None = None,
) -> str:
    q = (question or "").strip()

    if last_result_set_items:
        entity_type = (last_result_set_entity_type or "项").strip()
        item_text = "；".join(last_result_set_items[:20])

        if q.startswith(("还有", "另外", "除此之外", "还有没有", "还有别的")):
            return (
                f"已知{entity_type}集合如下：{item_text}。"
                f"请基于原问题继续判断，是否还有其他符合条件的{entity_type}。"
                f"原问题：{q}"
            )

        return (
            f"候选{entity_type}如下：{item_text}。"
            f"请只在这些候选项中回答：{q}"
        )

    anchor = infer_result_set_anchor(last_user_question, last_answer_type)

    if not anchor:
        return q

    if q.startswith(("都有哪些", "有哪些", "分别有哪些", "分别是哪些", "具体有哪些", "具体是哪些", "都是什么")):
        return f"{anchor}分别有哪些？请列出完整名单。"

    if q.startswith(("哪些", "哪个", "哪几个")):
        return f"{anchor}中，{q}"

    if q.startswith(("其中", "这里面")):
        return f"{anchor}里，{q}"

    return f"基于{anchor}，回答：{q}"

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

    patterns = [
        "多少文件", "多少个文件", "文件数量", "文档数量",
        "有哪些文件", "都有哪些文件", "文件清单",
        "有哪些文档", "都有哪些文档", "文档清单",
        "列一下", "列出来",
        "哪些类文档", "哪些类文件", "文档有哪些类", "文件有哪些类",
        "怎么分类", "如何分类", "分成哪些",
        "哪些格式", "文件格式", "文档格式",
        "最近更新", "最新文件", "最早文件",
        "最新文档", "最早文档",
        "占多大空间", "总共多大", "总大小", "总体积", "占用空间", "总容量",
    ]
    return any(p in q for p in patterns)


def looks_like_repo_time_question(question: str) -> bool:
    q = normalize_meta_question(question)
    return any(x in q for x in ["最早文件", "最早文档", "最新文件", "最新文档", "最近更新"])

def detect_dialog_event(question: str, state: ConversationState) -> DialogEvent:
    rs_match = looks_like_result_set_followup(question)
    has_state = state is not None
    last_answer_text = state.last_answer_text or state.last_answer_preview or ""
    has_last_answer = bool(last_answer_text)
    enum_like = last_turn_looks_like_enumeration(last_answer_text)
    last_answer_type = state.last_answer_type

    print(
        f"🧪 [result_set_followup判定] "
        f"rs_match={rs_match} | "
        f"has_state={has_state} | "
        f"has_last_answer={has_last_answer} | "
        f"enum_like={enum_like} | "
        f"last_answer_type={last_answer_type}"
    )
    prev_q = state.last_content_user_question
    prev_route = state.last_content_route
    last_topic = state.last_local_topic

    if looks_like_repo_time_question(question):
        return DialogEvent(name="repo_meta_request", route_hint="repo_meta")

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

    # === 7. 结果集追问（窄继承）
    is_result_set_answer = (
                                   state.last_answer_type in {"enumeration_company", "enumeration_file",
                                                              "enumeration_person"}
                           ) or enum_like

    if prev_route == "normal_retrieval" and rs_match and is_result_set_answer:
        return DialogEvent(name="result_set_followup", route_hint="normal_retrieval")
    # === 8. 内容追问继承
    if prev_route == "normal_retrieval" and is_context_dependent_question(question, state.last_effective_search_query):
        if prev_q:
            return DialogEvent(
                name="content_followup",
                route_hint="normal_retrieval",
                merged_query=f"{prev_q} {question}",
            )
        return DialogEvent(name="content_followup", route_hint="normal_retrieval")

    # === 9. 默认
    return DialogEvent(name="unknown", route_hint=None)

# =========================
# 状态转移（轻量）
# =========================

def apply_event_to_state(state: ConversationState, event: DialogEvent) -> ConversationState:
    new_state = ConversationState(
        mode=state.mode,
        last_user_question=state.last_user_question,
        last_route=state.last_route,
        last_local_topic=state.last_local_topic,
        last_answer_preview=state.last_answer_preview,

        last_content_user_question=state.last_content_user_question,
        last_content_route=state.last_content_route,
        last_content_topic=state.last_content_topic,

        last_effective_search_query=state.last_effective_search_query,
        last_answer_text=state.last_answer_text,
        last_answer_type=state.last_answer_type,
        last_result_set_query=state.last_result_set_query,

        last_result_set_items=state.last_result_set_items,
        last_result_set_entity_type=state.last_result_set_entity_type,
    )

    if event.route_hint == "repo_meta":
        new_state.mode = "repo_meta"
    elif event.route_hint == "normal_retrieval":
        new_state.mode = "content"
    elif event.name == "smalltalk":
        new_state.mode = "smalltalk"

    return new_state