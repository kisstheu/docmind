from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import re

def extract_result_set_from_answer(answer: str, entity_type: str = "文件") -> tuple[list[str], str]:
    items = re.findall(r"\d+\.\s*([^（]+?)\s*（", answer)
    items = [item.strip() for item in items if item.strip()]
    return items, entity_type

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

    pending_action_type: str | None = None
    pending_action_source_path: str | None = None
    pending_action_target_path: str | None = None
    pending_action_requested_text: str | None = None
    pending_action_preview: str | None = None


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
    r"^分别是关于什么",
    r"^分别是关于什么的",
    r"^各自是关于什么",
    r"^各自讲了什么",
    r"^分别讲了什么",
    r"^分别在说什么",
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
    r"^这两个",
    r"^这两个是",
    r"^这两个其实",
    r"^是不是说这两个",
    r"^这两个.*相同",
    r"^这两个.*一致",
    r"^它们是",
    r"^它们.*相同",
    r"^它们.*一致",
    r"^也就是说",
    r"^也就是说其实",
    r"^其实是",
    r"^其实.*一样",
    r"^其实.*相同",
    r"^其实.*一致",
    r"^也就是说.*相同",
    r"^也就是说.*一致",
    r"^也就是说.*一样",
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


RESULT_SET_COMPARISON_TERMS = [
    "不同", "区别", "差异", "异同",
    "相同", "一样", "一致",
    "对比", "比较",
]


RESULT_SET_GROUP_REF_TERMS = [
    "这几个", "这几份", "这两个", "这三", "这两",
    "这些", "它们", "上述", "前面", "上面", "其中",
]


def looks_like_result_set_followup(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False

    return (
        any(re.search(p, q) for p in RESULT_SET_FOLLOWUP_PATTERNS)
        or any(re.search(p, q) for p in RESULT_SET_CONTINUATION_PATTERNS)
    )


def looks_like_result_set_comparison_followup(question: str) -> bool:
    q = re.sub(r"\s+", "", (question or "").lower())
    if not q:
        return False

    if not any(term in q for term in RESULT_SET_COMPARISON_TERMS):
        return False

    if any(term in q for term in RESULT_SET_GROUP_REF_TERMS):
        return True

    return bool(re.search(r"(?:[一二两三四五六七八九十\d]+)(?:个|份|项|条)", q))


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


FILE_FOLLOWUP_STOP_TERMS = {
    "什么", "为何", "为什么", "怎么", "如何", "是否", "是不是",
    "哪个", "哪些", "哪几个", "几个", "三", "三个", "两", "两个", "几",
    "有啥", "有什么", "内容", "文件", "文档", "资料",
    "不同", "区别", "差异", "异同", "相同", "一样", "一致", "比较", "对比",
}


def _normalize_for_name_match(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "", (text or "").lower())


def _extract_question_focus_terms_for_files(question: str) -> list[str]:
    q = (question or "").lower()
    q = re.sub(r"[^\w\u4e00-\u9fa5]+", " ", q)
    raw_terms = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", q)

    terms: list[str] = []
    for term in raw_terms:
        t = term.strip()
        if not t or t in FILE_FOLLOWUP_STOP_TERMS:
            continue
        if re.fullmatch(r"\d+", t):
            continue
        if t not in terms:
            terms.append(t)
    return terms


def _narrow_result_set_files_by_question(items: list[str], question: str) -> list[str]:
    if not items:
        return items

    question_norm = _normalize_for_name_match(question)
    focus_terms = _extract_question_focus_terms_for_files(question)
    matched: list[str] = []
    for item in items:
        normalized_name = _normalize_for_name_match(item)
        # 1) 问题词直接命中文件名
        if focus_terms and any(term in normalized_name for term in focus_terms):
            matched.append(item)
            continue

        # 2) 反向匹配：文件名里的有效片段是否出现在问题中
        stem = re.sub(r"\.(txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx|png|jpg|jpeg|bmp|webp)$", "", item, flags=re.I)
        raw_parts = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", stem.lower())
        parts: list[str] = []
        for p in raw_parts:
            if p in FILE_FOLLOWUP_STOP_TERMS or re.fullmatch(r"\d+", p):
                continue
            if p not in parts:
                parts.append(p)

        matched_by_part = False
        for part in parts:
            if part in question_norm:
                matched_by_part = True
                break

            if re.fullmatch(r"[\u4e00-\u9fa5]{4,}", part):
                for width in (2, 3):
                    for i in range(0, len(part) - width + 1):
                        sub = part[i:i + width]
                        if sub in FILE_FOLLOWUP_STOP_TERMS:
                            continue
                        if sub in question_norm:
                            matched_by_part = True
                            break
                    if matched_by_part:
                        break
            if matched_by_part:
                break

        if matched_by_part:
            matched.append(item)

    # 至少保留 2 个候选，避免过度收窄
    return matched if len(matched) >= 2 else items


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
        candidate_items = list(last_result_set_items)
        if entity_type == "文件":
            candidate_items = _narrow_result_set_files_by_question(candidate_items, q)
        item_text = "；".join(candidate_items[:20])

        if q.startswith(("还有", "另外", "除此之外", "还有没有", "还有别的")):
            return (
                f"已知{entity_type}集合如下：{item_text}。"
                f"请基于原问题继续判断，是否还有其他符合条件的{entity_type}。"
                f"原问题：{q}"
            )

        if entity_type == "文件":
            return (
                f"已知文件如下：{item_text}。"
                f"请基于这些文件的内容回答：{q}"
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
    has_doc_word = any(x in q for x in ["文件", "文档", "资料"])

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
    if has_doc_word and any(x in q for x in ["列出", "列下", "列一下", "列出来", "罗列", "展开一下", "展开列一下"]):
        return True

    return False


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


def looks_like_repo_time_question(question: str, state: ConversationState | None = None) -> bool:
    q = normalize_meta_question(question)

    # 显式提到文件/文档 + 时间信号
    mentions_doc = any(x in q for x in ["文件", "文档", "资料", "pdf", "txt", "docx"])
    time_signals = [
        "最近", "最新", "最早", "最晚", "最旧",
        "最近更新", "最近修改", "更新时间",
        "修改时间", "创建时间", "时间", "日期",
        "时间最新", "时间最早",
    ]
    has_time_signal = any(x in q for x in time_signals)
    has_list_intent = any(x in q for x in ["有哪些", "有哪", "哪些", "哪几个", "列出", "列一下", "列出来"])
    has_explicit_time_axis = any(x in q for x in ["时间", "日期", "更新", "修改", "创建"])

    if has_time_signal and mentions_doc:
        return True

    if has_explicit_time_axis and has_time_signal:
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


def detect_dialog_event(question: str, state: ConversationState, logger) -> DialogEvent:
    rs_match = looks_like_result_set_followup(question)
    has_state = state is not None
    last_answer_text = state.last_answer_text or state.last_answer_preview or ""
    has_last_answer = bool(last_answer_text)
    enum_like = last_turn_looks_like_enumeration(last_answer_text)
    last_answer_type = state.last_answer_type

    logger.debug(
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


    if state.last_route == "repo_meta" and last_topic == "list_files" and is_list_format_modifier(question):
        return DialogEvent(name="repo_meta_request", route_hint="repo_meta")

    if looks_like_repo_time_question(question, state):
        return DialogEvent(name="repo_meta_request", route_hint="repo_meta")

    if looks_like_repo_size_consistency_followup(question, prev_q):
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

    if state.last_result_set_items and any(term in question.lower() for term in ["内容", "一样", "相同", "一致"]) and "文件" in question.lower():
        rs_match = True  # 强制设置匹配
    elif state.last_result_set_items and looks_like_result_set_comparison_followup(question):
        rs_match = True

    if prev_route in {"normal_retrieval", "repo_meta"} and rs_match and is_result_set_answer:
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

        pending_action_type=state.pending_action_type,
        pending_action_source_path=state.pending_action_source_path,
        pending_action_target_path=state.pending_action_target_path,
        pending_action_requested_text=state.pending_action_requested_text,
        pending_action_preview=state.pending_action_preview,
    )

    if event.route_hint == "repo_meta":
        new_state.mode = "repo_meta"
    elif event.route_hint == "normal_retrieval":
        new_state.mode = "content"
    elif event.name == "smalltalk":
        new_state.mode = "smalltalk"

    return new_state
