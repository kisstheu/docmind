from __future__ import annotations

from .comparison import (
    ConstraintAssessment,
    ConstraintStatus,
    JobRuleComparison,
)


_TITLE = "## 单 JD 显式规则比较"
_SECTION_ORDER = (
    (ConstraintStatus.MATCH, "明确符合"),
    (ConstraintStatus.CONFLICT, "明确冲突"),
    (ConstraintStatus.UNKNOWN, "信息缺失或需要确认"),
)
_FIELD_LABELS = {
    "salary": "薪资",
    "work_schedule": "工作制",
    "outsourcing": "外包",
    "onsite": "驻场",
    "location": "工作地点",
    "education": "学历",
    "experience": "经验年限",
}


def _render_assessment(assessment: ConstraintAssessment) -> list[str]:
    label = _FIELD_LABELS[assessment.field]
    evidence = (
        "；".join(assessment.jd_evidence)
        if assessment.jd_evidence
        else "JD未明确"
    )
    lines = [
        f"- {label}：{assessment.reason}",
        f"  JD证据：{evidence}",
    ]
    if assessment.confirmation_question is not None:
        lines.append(f"  建议确认：{assessment.confirmation_question}")
    return lines


def render_job_rule_comparison(result: JobRuleComparison) -> str:
    lines = [_TITLE]
    for status, heading in _SECTION_ORDER:
        lines.extend(("", f"### {heading}"))
        matching = tuple(
            assessment
            for assessment in result.assessments
            if assessment.status is status
        )
        if not matching:
            lines.append("- 无")
        else:
            for assessment in matching:
                lines.extend(_render_assessment(assessment))

    lines.extend(("", "### 技术要求原文"))
    if result.unassessed_technical_evidence:
        lines.extend(
            f"- {evidence}"
            for evidence in result.unassessed_technical_evidence
        )
    else:
        lines.append("- 无")
    lines.extend(
        (
            "",
            "> 仅比较调用方显式规则与 JD 明示硬条件；不生成投递决定。",
        )
    )
    return "\n".join(lines)
