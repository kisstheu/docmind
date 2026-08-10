from __future__ import annotations

import re


ANSWER_MODE_EVIDENCE = "evidence"
ANSWER_MODE_SYNTHESIS = "synthesis"
ANSWER_MODE_DECISION = "decision"
ANSWER_MODE_SELECTED_DETAIL = "selected_detail"

_COLLECTION_REFERENCES = (
    "这些", "那些", "上述", "它们", "这批", "该批", "这组", "这一组",
    "共同", "整体", "总体", "分别", "各自", "其中", "所有", "全部",
)
_SYNTHESIS_OPERATIONS = (
    "归纳", "汇总", "总结", "概括", "综合", "共同点", "共性", "规律",
    "分布", "趋势", "高频", "频率", "占比", "集中", "技术栈",
)
_SYNTHESIS_ASPECTS = (
    "要求", "条件", "能力", "技能", "标准", "限制", "约束", "风险",
    "特点", "特征", "差异", "共性", "流程", "步骤", "配置", "依赖",
)
_EVIDENCE_TARGETS = (
    "文件", "文档", "记录", "截图", "资料", "名称", "名字", "人名", "姓名",
    "位置", "出处", "来源", "第一个", "第一条", "第1个", "第1条",
)
_DETAIL_FOLLOWUPS = (
    "详细分析", "详细说", "详细讲", "详细说明",
    "展开分析", "展开说", "展开讲",
    "具体分析", "具体说明", "深入分析", "深入说明",
    "细说", "再分析", "分析一下", "分析下",
)

_OPEN_EVALUATION_TERMS = (
    "怎么样", "咋样", "好不好", "值不值得", "是否值得", "合不合适",
)
_EXPLICIT_EVALUATION_TERMS = (
    "评价一下", "评价下", "评估一下", "评估下", "判断一下", "判断下",
)
_FIT_OPERATORS = ("符合", "满足", "适合", "匹配")
_FIT_CRITERIA = ("条件", "要求", "标准", "目标", "需求", "偏好", "规则")


def _normalize(text: str) -> str:
    return re.sub(r"[，。！？、,.!?；;：:\s]+", "", (text or "").strip().lower())


def is_recommendation_request(question: str) -> bool:
    q = _normalize(question)
    if not q:
        return False
    if "推荐" in q:
        return True
    if re.search(r"(?:帮我|给我|替我)?(?:选|挑|选择|挑选)(?:出)?(?:一个|一份|一项|1个|1份|1项)", q):
        return True
    return bool(re.search(r"(?:哪个|哪一个|哪份|哪项).{0,10}(?:最适合|更适合|最匹配|更匹配|最好|最优)", q))


def is_comparison_or_ranking_request(question: str) -> bool:
    q = _normalize(question)
    if not q:
        return False
    if any(term in q for term in ("比较", "对比", "优劣", "取舍", "权衡")):
        return True
    if any(term in q for term in ("排序", "排名", "按匹配度", "从高到低", "从低到高")):
        return True
    return bool(re.search(r"(?:哪个|哪一个|哪份|哪项).{0,10}(?:更|最)(?:合适|适合|匹配|优)", q))


def is_collection_synthesis_request(question: str, *, has_collection_context: bool = False) -> bool:
    q = _normalize(question)
    if not q:
        return False
    if is_recommendation_request(q) or is_comparison_or_ranking_request(q):
        return False

    has_collection_signal = has_collection_context or any(term in q for term in _COLLECTION_REFERENCES)
    has_operation = any(term in q for term in _SYNTHESIS_OPERATIONS)
    if has_operation and has_collection_signal:
        return True

    asks_aspects = any(term in q for term in _SYNTHESIS_ASPECTS) and any(
        term in q for term in ("哪些", "什么", "有哪", "多少", "如何", "怎么样")
    )
    if asks_aspects and has_collection_signal:
        if any(target in q for target in _EVIDENCE_TARGETS) and any(
            locator in q for locator in ("哪份", "哪个", "哪些文件", "哪些文档", "在哪", "出处", "来源")
        ):
            return False
        return True
    return False


def is_selected_candidate_detail_request(question: str, *, has_selected_candidate: bool = False) -> bool:
    if not has_selected_candidate:
        return False
    return is_detail_explanation_request(question)


def is_detail_explanation_request(question: str) -> bool:
    q = _normalize(question)
    if not q or len(q) > 24:
        return False
    return any(term in q for term in _DETAIL_FOLLOWUPS)


def is_document_evaluation_request(question: str) -> bool:
    """Identify a domain-neutral request to evaluate one resolved document."""
    q = _normalize(question)
    if not q:
        return False
    if any(term in q for term in _OPEN_EVALUATION_TERMS):
        return True
    if any(term in q for term in _EXPLICIT_EVALUATION_TERMS):
        return True
    return any(operator in q for operator in _FIT_OPERATORS) and any(
        criterion in q for criterion in _FIT_CRITERIA
    )


def classify_answer_mode(
    question: str,
    *,
    has_collection_context: bool = False,
    has_selected_candidate: bool = False,
) -> str:
    if is_selected_candidate_detail_request(
        question,
        has_selected_candidate=has_selected_candidate,
    ):
        return ANSWER_MODE_SELECTED_DETAIL
    if is_recommendation_request(question) or is_comparison_or_ranking_request(question):
        return ANSWER_MODE_DECISION
    if is_collection_synthesis_request(
        question,
        has_collection_context=has_collection_context,
    ):
        return ANSWER_MODE_SYNTHESIS
    return ANSWER_MODE_EVIDENCE


def is_complex_answer_mode(mode: str) -> bool:
    return mode in {
        ANSWER_MODE_SYNTHESIS,
        ANSWER_MODE_DECISION,
        ANSWER_MODE_SELECTED_DETAIL,
    }


__all__ = [
    "ANSWER_MODE_DECISION",
    "ANSWER_MODE_EVIDENCE",
    "ANSWER_MODE_SELECTED_DETAIL",
    "ANSWER_MODE_SYNTHESIS",
    "classify_answer_mode",
    "is_collection_synthesis_request",
    "is_comparison_or_ranking_request",
    "is_complex_answer_mode",
    "is_detail_explanation_request",
    "is_document_evaluation_request",
    "is_recommendation_request",
    "is_selected_candidate_detail_request",
]
