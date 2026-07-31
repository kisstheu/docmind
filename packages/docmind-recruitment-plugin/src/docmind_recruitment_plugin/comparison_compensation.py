from __future__ import annotations

from decimal import Decimal
import re

from .comparison_contracts import ConstraintAssessment, ConstraintStatus
from .comparison_support import (
    _CONFLICT_MARKER,
    _decimal_text,
    _evidence,
    _unknown,
)


_SALARY_AMBIGUOUS = re.compile(
    r"面议|根据.{0,6}(?:能力|经验)|上不封顶|年薪|日薪|时薪|"
    r"\d+\s*薪|税前税后|另议"
)
_MONTHLY_SALARY_RANGE = re.compile(
    r"(?P<minimum>\d+(?:\.\d+)?)\s*(?:[kK千])?\s*"
    r"[-–—~～至到]\s*"
    r"(?P<maximum>\d+(?:\.\d+)?)\s*[kK千](?:元)?"
    r"(?:\s*/?\s*(?:月|每月))?"
)


def _assess_salary(
    fields: dict[str, str],
    rule: Decimal,
) -> ConstraintAssessment:
    evidence = _evidence(fields, "薪资")
    question = "该岗位可确认的月薪区间和实际 offer 下限是多少？"
    if not evidence:
        return _unknown(
            field="salary",
            rule_value=rule,
            evidence=(),
            reason="JD 未明确月薪区间。",
            question=question,
        )
    raw = evidence[0]
    if _CONFLICT_MARKER in raw or _SALARY_AMBIGUOUS.search(raw):
        return _unknown(
            field="salary",
            rule_value=rule,
            evidence=evidence,
            reason="JD 薪资表述含糊、冲突或不能无歧义换算为月薪区间。",
            question=question,
        )
    match = _MONTHLY_SALARY_RANGE.search(raw)
    if match is None:
        return _unknown(
            field="salary",
            rule_value=rule,
            evidence=evidence,
            reason="JD 薪资格式不能可靠标准化为 K/月区间。",
            question=question,
        )
    minimum = Decimal(match.group("minimum"))
    maximum = Decimal(match.group("maximum"))
    if minimum > maximum:
        return _unknown(
            field="salary",
            rule_value=rule,
            evidence=evidence,
            reason="JD 月薪区间上下限顺序不明确。",
            question=question,
        )
    jd_value = (minimum, maximum)
    if minimum >= rule:
        return ConstraintAssessment(
            field="salary",
            status=ConstraintStatus.MATCH,
            jd_value=jd_value,
            rule_value=rule,
            jd_evidence=evidence,
            reason=(
                f"JD 月薪下限 {_decimal_text(minimum)}K 不低于显式规则下限 "
                f"{_decimal_text(rule)}K。"
            ),
        )
    if maximum < rule:
        return ConstraintAssessment(
            field="salary",
            status=ConstraintStatus.CONFLICT,
            jd_value=jd_value,
            rule_value=rule,
            jd_evidence=evidence,
            reason=(
                f"JD 月薪上限 {_decimal_text(maximum)}K 低于显式规则下限 "
                f"{_decimal_text(rule)}K。"
            ),
        )
    return ConstraintAssessment(
        field="salary",
        status=ConstraintStatus.UNKNOWN,
        jd_value=jd_value,
        rule_value=rule,
        jd_evidence=evidence,
        reason="JD 月薪区间跨越显式规则下限，实际 offer 尚不确定。",
        confirmation_question=question,
    )
