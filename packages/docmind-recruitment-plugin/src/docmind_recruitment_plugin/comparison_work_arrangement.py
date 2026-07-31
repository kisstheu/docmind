from __future__ import annotations

import re

from .comparison_contracts import ConstraintAssessment, ConstraintStatus
from .comparison_support import _CONFLICT_MARKER, _evidence, _unknown


_NON_DOUBLE_WEEKEND = re.compile(
    r"单休|大小周|单双休|月休\s*(?:四|4|六|6)\s*天"
)
_DOUBLE_WEEKEND = re.compile(r"周末双休|双休")
_OUTSOURCING_NEGATIVE = re.compile(
    r"非外包|不是外包|不属于外包|不接受外包|非派遣|不是派遣|不派遣|"
    r"(?:与|和)?用人主体直接签约|直接与.{0,12}签订(?:劳动)?合同"
)
_OUTSOURCING_POSITIVE = re.compile(
    r"外包|派遣|第三方.{0,8}(?:签约|合同)"
)
_ONSITE_NEGATIVE = re.compile(
    r"不驻场|非驻场|无需驻场|不需要驻场|本公司办公|总部办公"
)
_ONSITE_POSITIVE = re.compile(
    r"长期驻场|常驻客户|驻客户.{0,8}(?:现场|办公)|客户现场办公"
)
_ONSITE_AMBIGUOUS = re.compile(
    r"偶尔|不定期|每月|出差|现场沟通|客户沟通|去客户处"
)
_LOCATION_AMBIGUOUS = re.compile(
    r"全国|待定|另行通知|就近分配|多地|远程|项目地点|根据项目"
)
_LOCATION_SEPARATOR = re.compile(r"[、,，;；/／或]")


def _assess_work_schedule(
    fields: dict[str, str],
    require_double_weekends: bool,
) -> ConstraintAssessment:
    evidence = _evidence(fields, "工作制", "加班或大小周")
    question = "该岗位是否固定周末双休？"
    if not evidence:
        return _unknown(
            field="work_schedule",
            rule_value=require_double_weekends,
            evidence=(),
            reason="JD 未明确工作制。",
            question=question,
        )
    combined = "；".join(evidence)
    if _CONFLICT_MARKER in combined:
        return _unknown(
            field="work_schedule",
            rule_value=require_double_weekends,
            evidence=evidence,
            reason="JD 工作制原文存在冲突，不能可靠判断。",
            question=question,
        )
    is_non_double = _NON_DOUBLE_WEEKEND.search(combined) is not None
    is_double = _DOUBLE_WEEKEND.search(combined) is not None
    if require_double_weekends and is_non_double:
        return ConstraintAssessment(
            field="work_schedule",
            status=ConstraintStatus.CONFLICT,
            jd_value="not_double_weekends",
            rule_value=True,
            jd_evidence=evidence,
            reason="JD 明确的工作制不是固定双休，与显式规则冲突。",
        )
    if require_double_weekends and is_double:
        return ConstraintAssessment(
            field="work_schedule",
            status=ConstraintStatus.MATCH,
            jd_value="double_weekends",
            rule_value=True,
            jd_evidence=evidence,
            reason="JD 明确为双休，满足显式规则。",
        )
    if not require_double_weekends and (is_non_double or is_double):
        return ConstraintAssessment(
            field="work_schedule",
            status=ConstraintStatus.MATCH,
            jd_value=(
                "not_double_weekends" if is_non_double else "double_weekends"
            ),
            rule_value=False,
            jd_evidence=evidence,
            reason="JD 已明确工作制，调用方未要求固定双休。",
        )
    return _unknown(
        field="work_schedule",
        rule_value=require_double_weekends,
        evidence=evidence,
        reason="JD 工作制表述不足以确认是否固定双休。",
        question=question,
    )


def _assess_outsourcing(
    fields: dict[str, str],
    allow_outsourcing: bool,
) -> ConstraintAssessment:
    evidence = _evidence(fields, "外包")
    question = "该岗位是否为外包、派遣或第三方签约？"
    if not evidence:
        return _unknown(
            field="outsourcing",
            rule_value=allow_outsourcing,
            evidence=(),
            reason="JD 未明确是否外包。",
            question=question,
        )
    raw = evidence[0]
    if _CONFLICT_MARKER in raw:
        return _unknown(
            field="outsourcing",
            rule_value=allow_outsourcing,
            evidence=evidence,
            reason="JD 外包信息原文存在冲突。",
            question=question,
        )
    compact = "".join(raw.split())
    is_negative = (
        compact in {"否", "不是", "非外包"}
        or _OUTSOURCING_NEGATIVE.search(raw) is not None
    )
    is_positive = (
        compact in {"是", "外包"}
        or (
            _OUTSOURCING_POSITIVE.search(raw) is not None
            and not is_negative
        )
    )
    if not (is_negative or is_positive):
        return _unknown(
            field="outsourcing",
            rule_value=allow_outsourcing,
            evidence=evidence,
            reason="JD 原文不足以可靠确认签约主体或外包性质。",
            question=question,
        )
    is_outsourcing = is_positive
    status = (
        ConstraintStatus.MATCH
        if allow_outsourcing or not is_outsourcing
        else ConstraintStatus.CONFLICT
    )
    reason = (
        "JD 明确为外包、派遣或第三方签约，调用方允许外包。"
        if is_outsourcing and allow_outsourcing
        else "JD 明确为外包、派遣或第三方签约，与不接受外包的规则冲突。"
        if is_outsourcing
        else "JD 明确为非外包或直接与用人主体签约，满足显式规则。"
    )
    return ConstraintAssessment(
        field="outsourcing",
        status=status,
        jd_value=is_outsourcing,
        rule_value=allow_outsourcing,
        jd_evidence=evidence,
        reason=reason,
    )


def _assess_onsite(
    fields: dict[str, str],
    allow_onsite: bool,
) -> ConstraintAssessment:
    evidence = _evidence(fields, "驻场")
    question = "该岗位是否需要长期驻客户现场办公？"
    if not evidence:
        return _unknown(
            field="onsite",
            rule_value=allow_onsite,
            evidence=(),
            reason="JD 未明确是否长期驻场。",
            question=question,
        )
    raw = evidence[0]
    if _CONFLICT_MARKER in raw or _ONSITE_AMBIGUOUS.search(raw):
        return _unknown(
            field="onsite",
            rule_value=allow_onsite,
            evidence=evidence,
            reason="JD 只描述现场沟通、偶发安排或冲突信息，不能确认长期驻场。",
            question=question,
        )
    compact = "".join(raw.split())
    is_negative = (
        compact in {"否", "不是", "不驻场"}
        or _ONSITE_NEGATIVE.search(raw) is not None
    )
    is_positive = (
        compact in {"是", "驻场"}
        or (
            _ONSITE_POSITIVE.search(raw) is not None
            and not is_negative
        )
    )
    if not (is_negative or is_positive):
        return _unknown(
            field="onsite",
            rule_value=allow_onsite,
            evidence=evidence,
            reason="JD 原文不足以确认是否长期驻场。",
            question=question,
        )
    is_onsite = is_positive
    status = (
        ConstraintStatus.MATCH
        if allow_onsite or not is_onsite
        else ConstraintStatus.CONFLICT
    )
    reason = (
        "JD 明确需要长期驻场，调用方允许驻场。"
        if is_onsite and allow_onsite
        else "JD 明确需要长期驻场，与不接受驻场的规则冲突。"
        if is_onsite
        else "JD 明确不驻场或在本公司办公，满足显式规则。"
    )
    return ConstraintAssessment(
        field="onsite",
        status=status,
        jd_value=is_onsite,
        rule_value=allow_onsite,
        jd_evidence=evidence,
        reason=reason,
    )


def _normalize_location(value: str) -> str:
    normalized = "".join(value.casefold().split())
    if normalized.endswith(("省", "市", "区", "县")):
        normalized = normalized[:-1]
    return normalized


def _assess_location(
    fields: dict[str, str],
    allowed_locations: tuple[str, ...],
) -> ConstraintAssessment:
    evidence = _evidence(fields, "工作地点")
    question = "该岗位唯一、确定的办公地点是什么？"
    if not evidence:
        return _unknown(
            field="location",
            rule_value=allowed_locations,
            evidence=(),
            reason="JD 未明确工作地点。",
            question=question,
        )
    raw = evidence[0]
    if (
        _CONFLICT_MARKER in raw
        or _LOCATION_AMBIGUOUS.search(raw)
        or _LOCATION_SEPARATOR.search(raw)
    ):
        return _unknown(
            field="location",
            rule_value=allowed_locations,
            evidence=evidence,
            reason="JD 地点为多选、待定、远程或其他不唯一表述。",
            question=question,
        )
    normalized = _normalize_location(raw)
    if not normalized:
        return _unknown(
            field="location",
            rule_value=allowed_locations,
            evidence=evidence,
            reason="JD 工作地点无法可靠规范化。",
            question=question,
        )
    allowed = tuple(_normalize_location(value) for value in allowed_locations)
    status = (
        ConstraintStatus.MATCH
        if normalized in allowed
        else ConstraintStatus.CONFLICT
    )
    reason = (
        "JD 唯一明确地点属于调用方显式允许列表。"
        if status is ConstraintStatus.MATCH
        else "JD 唯一明确地点不属于调用方显式允许列表。"
    )
    return ConstraintAssessment(
        field="location",
        status=status,
        jd_value=normalized,
        rule_value=allowed_locations,
        jd_evidence=evidence,
        reason=reason,
    )
