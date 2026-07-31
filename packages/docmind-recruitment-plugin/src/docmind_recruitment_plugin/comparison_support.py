from __future__ import annotations

from decimal import Decimal

from .comparison_contracts import ConstraintAssessment, ConstraintStatus
from .extraction import ExtractionResult


_CONFLICT_MARKER = "原文存在冲突"


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _field_map(result: ExtractionResult) -> dict[str, str]:
    return dict(result.fields)


def _evidence(fields: dict[str, str], *names: str) -> tuple[str, ...]:
    return tuple(fields[name] for name in names if name in fields)


def _unknown(
    *,
    field: str,
    rule_value: object,
    evidence: tuple[str, ...],
    reason: str,
    question: str,
) -> ConstraintAssessment:
    return ConstraintAssessment(
        field=field,
        status=ConstraintStatus.UNKNOWN,
        jd_value=None,
        rule_value=rule_value,
        jd_evidence=evidence,
        reason=reason,
        confirmation_question=question,
    )
