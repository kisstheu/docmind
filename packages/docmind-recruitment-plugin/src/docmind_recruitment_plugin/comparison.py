from __future__ import annotations

from .comparison_compensation import _assess_salary
from .comparison_contracts import (
    ConstraintAssessment,
    ConstraintStatus,
    EducationLevel,
    JobRuleComparison,
    JobSearchRules,
)
from .comparison_qualifications import (
    _assess_education,
    _assess_experience,
)
from .comparison_support import _field_map
from .comparison_work_arrangement import (
    _assess_location,
    _assess_onsite,
    _assess_outsourcing,
    _assess_work_schedule,
)
from .extraction import ExtractionResult, extract_constraints


def _technical_evidence(fields: dict[str, str]) -> tuple[str, ...]:
    raw = fields.get("技术要求")
    if raw is None:
        return ()
    return tuple(
        fragment.strip()
        for fragment in raw.split("；")
        if fragment.strip()
    )


def compare_job_search_rules(
    extracted: ExtractionResult,
    rules: JobSearchRules,
) -> JobRuleComparison:
    fields = _field_map(extracted)
    assessments: list[ConstraintAssessment] = []
    if rules.minimum_monthly_salary_k is not None:
        assessments.append(
            _assess_salary(fields, rules.minimum_monthly_salary_k)
        )
    if rules.require_double_weekends is not None:
        assessments.append(
            _assess_work_schedule(fields, rules.require_double_weekends)
        )
    if rules.allow_outsourcing is not None:
        assessments.append(
            _assess_outsourcing(fields, rules.allow_outsourcing)
        )
    if rules.allow_onsite is not None:
        assessments.append(_assess_onsite(fields, rules.allow_onsite))
    if rules.allowed_locations:
        assessments.append(
            _assess_location(fields, rules.allowed_locations)
        )
    if rules.candidate_education_level is not None:
        assessments.append(
            _assess_education(fields, rules.candidate_education_level)
        )
    if rules.candidate_relevant_years is not None:
        assessments.append(
            _assess_experience(fields, rules.candidate_relevant_years)
        )
    return JobRuleComparison(
        assessments=tuple(assessments),
        unassessed_technical_evidence=_technical_evidence(fields),
    )


def extract_and_compare(
    query: str,
    rules: JobSearchRules,
) -> JobRuleComparison | None:
    extracted = extract_constraints(query)
    if extracted is None:
        return None
    return compare_job_search_rules(extracted, rules)
