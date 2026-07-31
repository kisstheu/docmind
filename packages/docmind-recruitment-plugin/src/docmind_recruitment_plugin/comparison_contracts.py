from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class EducationLevel(str, Enum):
    UNRESTRICTED = "unrestricted"
    SECONDARY = "secondary"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"


class ConstraintStatus(str, Enum):
    MATCH = "match"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class JobSearchRules:
    minimum_monthly_salary_k: Decimal | None = None
    require_double_weekends: bool | None = None
    allow_outsourcing: bool | None = None
    allow_onsite: bool | None = None
    allowed_locations: tuple[str, ...] = ()
    candidate_education_level: EducationLevel | None = None
    candidate_relevant_years: Decimal | None = None

    def __post_init__(self) -> None:
        _validate_decimal(
            "minimum_monthly_salary_k",
            self.minimum_monthly_salary_k,
        )
        _validate_decimal(
            "candidate_relevant_years",
            self.candidate_relevant_years,
        )
        for name in (
            "require_double_weekends",
            "allow_outsourcing",
            "allow_onsite",
        ):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool or None")
        if not isinstance(self.allowed_locations, tuple):
            raise TypeError("allowed_locations must be a tuple")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in self.allowed_locations
        ):
            raise ValueError("allowed_locations must contain non-empty strings")
        if (
            self.candidate_education_level is not None
            and not isinstance(self.candidate_education_level, EducationLevel)
        ):
            raise TypeError(
                "candidate_education_level must be EducationLevel or None"
            )


@dataclass(frozen=True)
class ConstraintAssessment:
    field: str
    status: ConstraintStatus
    jd_value: object | None
    rule_value: object | None
    jd_evidence: tuple[str, ...]
    reason: str
    confirmation_question: str | None = None


@dataclass(frozen=True)
class JobRuleComparison:
    assessments: tuple[ConstraintAssessment, ...]
    unassessed_technical_evidence: tuple[str, ...] = ()

    @property
    def match_count(self) -> int:
        return sum(
            assessment.status is ConstraintStatus.MATCH
            for assessment in self.assessments
        )

    @property
    def conflict_count(self) -> int:
        return sum(
            assessment.status is ConstraintStatus.CONFLICT
            for assessment in self.assessments
        )

    @property
    def unknown_count(self) -> int:
        return sum(
            assessment.status is ConstraintStatus.UNKNOWN
            for assessment in self.assessments
        )


def _validate_decimal(name: str, value: Decimal | None) -> None:
    if value is None:
        return
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be Decimal or None")
    if not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
