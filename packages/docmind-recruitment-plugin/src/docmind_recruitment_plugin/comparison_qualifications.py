from __future__ import annotations

from decimal import Decimal
import re

from .comparison_contracts import (
    ConstraintAssessment,
    ConstraintStatus,
    EducationLevel,
)
from .comparison_support import (
    _CONFLICT_MARKER,
    _decimal_text,
    _evidence,
    _unknown,
)


_EDUCATION_SOFT = re.compile(r"优先|加分|可放宽|优秀者|条件优秀|能力突出")
_EDUCATION_EXTRA_QUALIFIER = re.compile(
    r"统招|全日制|非全日制|第一学历|学位"
)
_EDUCATION_PATTERNS = (
    (EducationLevel.DOCTORATE, re.compile(r"博士")),
    (EducationLevel.MASTER, re.compile(r"硕士|研究生")),
    (EducationLevel.BACHELOR, re.compile(r"本科")),
    (EducationLevel.ASSOCIATE, re.compile(r"大专|专科")),
    (EducationLevel.SECONDARY, re.compile(r"中专|高中")),
)
_EDUCATION_RANK = {
    EducationLevel.UNRESTRICTED: 0,
    EducationLevel.SECONDARY: 1,
    EducationLevel.ASSOCIATE: 2,
    EducationLevel.BACHELOR: 3,
    EducationLevel.MASTER: 4,
    EducationLevel.DOCTORATE: 5,
}
_EXPERIENCE_SOFT = re.compile(r"优先|加分|经验丰富|有经验者|经验不限")
_EXPERIENCE_MAXIMUM_ONLY = re.compile(r"以下|不超过|至多")
_EXPERIENCE_APPROXIMATE = re.compile(
    r"(?:大约|约)\s*\d+(?:\.\d+)?\s*年|"
    r"\d+(?:\.\d+)?\s*年\s*(?:左右|上下)"
)
_EXPERIENCE_RANGE = re.compile(
    r"(?P<minimum>\d+(?:\.\d+)?)\s*"
    r"[-–—~～至到]\s*"
    r"(?P<maximum>\d+(?:\.\d+)?)\s*年"
)
_EXPERIENCE_MINIMUM = re.compile(
    r"(?:至少|不低于)?\s*(?P<minimum>\d+(?:\.\d+)?)\s*年(?:及?以上)?"
)


def _assess_education(
    fields: dict[str, str],
    candidate_level: EducationLevel,
) -> ConstraintAssessment:
    evidence = _evidence(fields, "学历")
    question = "该学历是硬性最低要求吗，是否还有限定的学历类型？"
    if not evidence:
        return _unknown(
            field="education",
            rule_value=candidate_level,
            evidence=(),
            reason="JD 未明确学历要求。",
            question=question,
        )
    raw = evidence[0]
    if (
        _CONFLICT_MARKER in raw
        or _EDUCATION_SOFT.search(raw)
        or _EDUCATION_EXTRA_QUALIFIER.search(raw)
    ):
        return _unknown(
            field="education",
            rule_value=candidate_level,
            evidence=evidence,
            reason="JD 学历表述是偏好、可放宽、冲突或带有额外学历类型限定。",
            question=question,
        )
    if "不限" in raw:
        minimum = EducationLevel.UNRESTRICTED
    else:
        levels = tuple(
            level for level, pattern in _EDUCATION_PATTERNS if pattern.search(raw)
        )
        if len(levels) != 1:
            return _unknown(
                field="education",
                rule_value=candidate_level,
                evidence=evidence,
                reason="JD 学历最低层级不能可靠确定。",
                question=question,
            )
        minimum = levels[0]
    status = (
        ConstraintStatus.MATCH
        if _EDUCATION_RANK[candidate_level] >= _EDUCATION_RANK[minimum]
        else ConstraintStatus.CONFLICT
    )
    reason = (
        "调用方显式提供的候选学历达到 JD 硬性最低学历。"
        if status is ConstraintStatus.MATCH
        else "调用方显式提供的候选学历低于 JD 硬性最低学历。"
    )
    return ConstraintAssessment(
        field="education",
        status=status,
        jd_value=minimum,
        rule_value=candidate_level,
        jd_evidence=evidence,
        reason=reason,
    )


def _assess_experience(
    fields: dict[str, str],
    candidate_years: Decimal,
) -> ConstraintAssessment:
    evidence = _evidence(fields, "经验")
    question = "该岗位硬性要求的最低相关经验年限是多少？"
    if not evidence:
        return _unknown(
            field="experience",
            rule_value=candidate_years,
            evidence=(),
            reason="JD 未明确硬性最低经验年限。",
            question=question,
        )
    raw = evidence[0]
    if _EXPERIENCE_APPROXIMATE.search(raw):
        return _unknown(
            field="experience",
            rule_value=candidate_years,
            evidence=evidence,
            reason="JD 只给出近似年限，不能作为明确的硬性最低经验要求。",
            question=question,
        )
    if (
        _CONFLICT_MARKER in raw
        or _EXPERIENCE_SOFT.search(raw)
        or _EXPERIENCE_MAXIMUM_ONLY.search(raw)
    ):
        return _unknown(
            field="experience",
            rule_value=candidate_years,
            evidence=evidence,
            reason="JD 经验表述是偏好、含糊、仅给上限或存在原文冲突。",
            question=question,
        )
    range_match = _EXPERIENCE_RANGE.search(raw)
    if range_match is not None:
        minimum = Decimal(range_match.group("minimum"))
        maximum = Decimal(range_match.group("maximum"))
        if minimum > maximum:
            return _unknown(
                field="experience",
                rule_value=candidate_years,
                evidence=evidence,
                reason="JD 经验区间上下限顺序不明确。",
                question=question,
            )
    else:
        minimum_match = _EXPERIENCE_MINIMUM.search(raw)
        if minimum_match is None:
            return _unknown(
                field="experience",
                rule_value=candidate_years,
                evidence=evidence,
                reason="JD 硬性最低经验年限不能可靠解析。",
                question=question,
            )
        minimum = Decimal(minimum_match.group("minimum"))
        maximum = None
    status = (
        ConstraintStatus.MATCH
        if candidate_years >= minimum
        else ConstraintStatus.CONFLICT
    )
    reason = (
        f"调用方显式提供的相关经验 {_decimal_text(candidate_years)} 年达到 JD 最低要求 "
        f"{_decimal_text(minimum)} 年。"
        if status is ConstraintStatus.MATCH
        else f"调用方显式提供的相关经验 {_decimal_text(candidate_years)} 年低于 JD 最低要求 "
        f"{_decimal_text(minimum)} 年。"
    )
    return ConstraintAssessment(
        field="experience",
        status=status,
        jd_value=(minimum, maximum),
        rule_value=candidate_years,
        jd_evidence=evidence,
        reason=reason,
    )
