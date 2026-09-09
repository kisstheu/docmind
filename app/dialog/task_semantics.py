from __future__ import annotations

import os
import re

from ai.table_presentation import StructuredTable, refine_structured_table
from app.dialog.repo_meta_rules import is_explicit_corpus_content_enumeration_request


ANSWER_MODE_EVIDENCE = "evidence"
ANSWER_MODE_SYNTHESIS = "synthesis"
ANSWER_MODE_DECISION = "decision"
ANSWER_MODE_SELECTED_DETAIL = "selected_detail"

_HIGH_CONFIDENCE_FOLLOWUP_CLAUSE_SPLIT = re.compile(
    r"[，,；;]|(?:并且|而且|同时|另外|然后|以及|还要|还得)"
)
_CONDITIONAL_FOLLOWUP_CLAUSE_SPLIT = re.compile(r"并(?!列)|且|再")
_PRESENTATION_DISCOURSE_PREFIX = r"(?:那|那么|那就)?"
_PRESENTATION_POLITE_PREFIX = (
    r"(?:(?:可以|能否|能不能|请|请你|麻烦|麻烦你|帮我|替我|给我))*"
)
_PREVIOUS_ANSWER_REFERENCE = (
    r"(?:把)?(?:"
    r"(?:刚才|上一轮|上轮|前面|上面|上述)(?:的)?(?:回答|内容|结果)?"
    r"|(?:这个|这段|这些|那些|它|它们)(?:回答|内容|结果)?"
    r")"
)
_PRESENTATION_TARGET = (
    r"(?:markdown)?(?:表格|列表|清单|要点|分点)"
    r"|(?:一|二|两|三|四|五|六|七|八|九|十|\d+)列"
)
_PRESENTATION_OPERATION = (
    r"(?:合并|(?:重新|再|合并|并列)?"
    r"(?:整理|梳理|改|换|转换|转|做|制作|弄|列|表示|呈现|输出|来))"
)
_PRESENTATION_SOFT_SUFFIX = (
    r"(?:(?:一下|下|看看|看下|看一下))?(?:吧|吗|呢|呀|啊)?"
)
_PRESENTATION_LAYOUT_SPEC = (
    rf"(?:类别|类型|时间顺序|先后顺序|字母顺序|名称|编号|序号|{_PRESENTATION_TARGET})"
)

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

_EXPLANATORY_FOLLOWUP_TERMS = (
    "什么意思", "怎么理解", "如何理解", "到底是啥", "到底是什么",
    "具体讲讲", "具体说说", "讲清楚", "说清楚", "解释一下", "解释下",
    "干嘛的", "做什么用", "有什么用", "有什么作用", "怎么起作用",
    "为什么", "为何", "原理", "和刚才", "有什么关系",
)

_EVIDENCE_LOOKUP_TERMS = (
    "哪份文件", "哪个文件", "哪些文件", "哪篇文档", "哪个文档", "哪些文档",
    "在哪一页", "第几页", "哪一页", "原文", "来源", "出处", "哪句",
    "哪里写", "哪儿写", "还有别的文件", "是否提到", "有没有提到",
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


def _has_answer_presentation_intent(text: str) -> bool:
    q = _normalize(text)
    if not q:
        return False
    prefix = rf"{_PRESENTATION_DISCOURSE_PREFIX}{_PRESENTATION_POLITE_PREFIX}"
    referent = rf"(?:{_PREVIOUS_ANSWER_REFERENCE})?"
    target = rf"(?:{_PRESENTATION_TARGET})"
    target_form = rf"(?:个|一张|一种)?{target}(?:的)?(?:形式|格式|方式)?"
    operation = _PRESENTATION_OPERATION
    suffix = _PRESENTATION_SOFT_SUFFIX
    patterns = (
        rf"{prefix}{referent}{operation}(?:成|为)?{target_form}{suffix}",
        rf"{prefix}{referent}{target_form}{operation}{suffix}",
        rf"{prefix}{referent}用{target_form}(?:表示|呈现|输出)?{suffix}",
        rf"{prefix}{referent}{target_form}{suffix}",
        rf"{prefix}{referent}(?:简短|简单|精简)(?:一点|一些|些|点)?{suffix}",
        rf"{prefix}{referent}(?:更简短|更简单|更精简)(?:一点|一些|些|点)?{suffix}",
        rf"{prefix}{referent}换(?:一种|个)格式{suffix}",
        rf"{prefix}{referent}按{_PRESENTATION_LAYOUT_SPEC}(?:排|排列|排序|整理){suffix}",
    )
    return any(re.fullmatch(pattern, q) for pattern in patterns)


def _split_followup_clauses(question: str) -> list[str]:
    """Split only at connectors that occupy a clause boundary."""
    clauses: list[str] = []
    for segment in _HIGH_CONFIDENCE_FOLLOWUP_CLAUSE_SPLIT.split(
        (question or "").lower()
    ):
        segment = segment.strip()
        if not segment:
            continue

        start = 0
        for match in _CONDITIONAL_FOLLOWUP_CLAUSE_SPLIT.finditer(segment):
            preceding = segment[start:match.start()].strip()
            if preceding and _has_answer_presentation_intent(preceding):
                clauses.append(preceding)
                start = match.end()

        remainder = segment[start:].strip()
        if remainder:
            clauses.append(remainder)
    return clauses


def is_answer_presentation_followup(
    question: str,
    *,
    has_previous_answer: bool,
    current_table: StructuredTable | None = None,
) -> bool:
    """Identify a previous-answer-only presentation transform.

    Presentation clauses may be implicit references when an answer exists. A
    second non-presentation clause is treated as a new factual demand and must
    continue through retrieval.
    """
    if not has_previous_answer:
        return False

    if refine_structured_table(current_table, question).valid:
        return True

    clauses = _split_followup_clauses(question)
    return bool(clauses) and all(
        _has_answer_presentation_intent(clause) for clause in clauses
    )


def is_table_presentation_request(question: str) -> bool:
    """Identify the table execution subtype after pure-presentation routing."""
    return "表格" in _normalize(question)


def is_recommendation_request(question: str) -> bool:
    q = _normalize(question)
    if not q:
        return False
    # A polar question about a recommendation does not request a candidate.
    # Keep explicit recipients/quantities and any separate selection request.
    selection_text = re.sub(
        r"(?:是否|有没有|有无)推荐(?!给|我|我们|一个|一份|一项|1个|1份|1项)",
        "",
        q,
    )
    if "推荐" in selection_text:
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


def needs_multi_source_decision_delivery(question: str, source_paths) -> bool:
    """Delivery scope follows comparison intent and available evidence, not selection count."""
    if os.getenv("DOCMIND_MULTI_SOURCE_DECISION_DELIVERY", "1").strip().lower() in {
        "0", "false", "off",
    }:
        return False
    q = _normalize(question)
    # An action request can ask for a relative priority without saying “compare”.
    # Require both a collection and an interrogative priority, not domain nouns
    # or a factual mention of priority in a single object's documentation.
    collection_priority = any(term in q for term in (*_COLLECTION_REFERENCES, "一批", "几个")) and bool(
        re.search(r"(?:优先|先)(?:联系|选择|考虑|采用|确认|处理)(?:谁|哪)", q)
        or re.search(r"(?:谁|哪个|哪一个|哪项).{0,8}(?:优先|先)(?:联系|选择|考虑|采用|确认|处理)", q)
    )
    return (is_comparison_or_ranking_request(q) or collection_priority) and len(set(source_paths)) > 1


def is_collection_synthesis_request(question: str, *, has_collection_context: bool = False) -> bool:
    q = _normalize(question)
    if not q:
        return False
    if is_recommendation_request(q) or is_comparison_or_ranking_request(q):
        return False
    if is_explicit_corpus_content_enumeration_request(question):
        return True

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


def is_subjectless_collection_synthesis_request(question: str) -> bool:
    """Identify a synthesis operation whose omitted subject must come from context."""
    q = _normalize(question)
    if not q:
        return False
    return bool(
        re.fullmatch(
            r"(?:请|帮我|麻烦)?"
            r"(?:归纳|汇总|总结|概括|综合)(?:一下|下)?"
            r"(?:都|分别|各自)?(?:有)?"
            r"(?:哪些|哪几个|哪几条|哪几项|哪几种|有什么|有啥).+",
            q,
        )
    )


def is_selected_candidate_detail_request(question: str, *, has_selected_candidate: bool = False) -> bool:
    if not has_selected_candidate:
        return False
    return is_detail_explanation_request(question)


def is_detail_explanation_request(question: str) -> bool:
    q = _normalize(question)
    if not q or len(q) > 24:
        return False
    return any(term in q for term in _DETAIL_FOLLOWUPS)


def is_explanatory_followup_request(question: str) -> bool:
    """Identify a domain-neutral request to explain or clarify existing content."""
    q = _normalize(question)
    if not q or len(q) > 40:
        return False
    if any(term in q for term in _EVIDENCE_LOOKUP_TERMS):
        return False
    if is_detail_explanation_request(question):
        return True
    if any(term in q for term in _EXPLANATORY_FOLLOWUP_TERMS):
        return True
    return bool(
        re.search(r"^(?:什么是).+", q)
        or re.search(r".+(?:又|到底)?(?:是啥|是什么)$", q)
    )


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
    has_collection_open_enumeration: bool = False,
    has_selected_candidate: bool = False,
) -> str:
    if is_selected_candidate_detail_request(
        question,
        has_selected_candidate=has_selected_candidate,
    ):
        return ANSWER_MODE_SELECTED_DETAIL
    if is_recommendation_request(question) or is_comparison_or_ranking_request(question):
        return ANSWER_MODE_DECISION
    if has_collection_open_enumeration:
        return ANSWER_MODE_SYNTHESIS
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
    "is_explanatory_followup_request",
    "is_answer_presentation_followup",
    "is_table_presentation_request",
    "is_recommendation_request",
    "is_selected_candidate_detail_request",
    "is_subjectless_collection_synthesis_request",
]
