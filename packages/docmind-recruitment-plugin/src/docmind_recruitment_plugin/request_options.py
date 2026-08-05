from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from docmind_domain_sdk import JsonValue

from .comparison_contracts import EducationLevel, JobSearchRules


OPTIONS_SCHEMA_VERSION = "1.0"
INVALID_OPTIONS_CODE = "recruitment.options.invalid_payload"
INVALID_OPTIONS_CATEGORY = "invalid_request_options"

_ERROR_TEXT = f"{INVALID_OPTIONS_CODE}: {INVALID_OPTIONS_CATEGORY}"
_NAMESPACE_FIELDS = frozenset(("schema_version", "explicit_rules"))
_RULE_FIELDS = frozenset(
    (
        "minimum_monthly_salary_k",
        "require_double_weekends",
        "allow_outsourcing",
        "allow_onsite",
        "allowed_locations",
        "candidate_education_level",
        "candidate_relevant_years",
    )
)
_NUMBER_FIELDS = (
    "minimum_monthly_salary_k",
    "candidate_relevant_years",
)
_BOOLEAN_FIELDS = (
    "require_double_weekends",
    "allow_outsourcing",
    "allow_onsite",
)


class RecruitmentOptionsError(ValueError):
    code = INVALID_OPTIONS_CODE
    category = INVALID_OPTIONS_CATEGORY

    def __init__(self) -> None:
        super().__init__(_ERROR_TEXT)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"code={self.code!r}, category={self.category!r})"
        )


def _invalid() -> None:
    raise RecruitmentOptionsError


def _decimal_number(value: JsonValue) -> Decimal:
    if type(value) not in (int, float):
        _invalid()
    converted = Decimal(str(value))
    if not converted.is_finite() or converted < 0:
        _invalid()
    return converted


def _locations(value: JsonValue) -> tuple[str, ...]:
    if type(value) is not list:
        _invalid()
    if any(type(item) is not str or not item.strip() for item in value):
        _invalid()
    return tuple(value)


def _education(value: JsonValue) -> EducationLevel:
    if type(value) is not str:
        _invalid()
    try:
        return EducationLevel(value)
    except ValueError:
        _invalid()


def parse_recruitment_request_options(
    options: Mapping[str, JsonValue],
) -> JobSearchRules | None:
    from .plugin import PLUGIN_ID

    if not isinstance(options, Mapping):
        _invalid()
    if PLUGIN_ID not in options:
        return None

    namespace = options[PLUGIN_ID]
    if not isinstance(namespace, Mapping):
        _invalid()
    if set(namespace) != _NAMESPACE_FIELDS:
        _invalid()
    if namespace["schema_version"] != OPTIONS_SCHEMA_VERSION or type(
        namespace["schema_version"]
    ) is not str:
        _invalid()

    explicit_rules = namespace["explicit_rules"]
    if not isinstance(explicit_rules, Mapping):
        _invalid()
    if not set(explicit_rules).issubset(_RULE_FIELDS):
        _invalid()

    values: dict[str, object] = {}
    for field in _NUMBER_FIELDS:
        if field in explicit_rules:
            values[field] = _decimal_number(explicit_rules[field])
    for field in _BOOLEAN_FIELDS:
        if field in explicit_rules:
            value = explicit_rules[field]
            if type(value) is not bool:
                _invalid()
            values[field] = value
    if "allowed_locations" in explicit_rules:
        values["allowed_locations"] = _locations(
            explicit_rules["allowed_locations"]
        )
    if "candidate_education_level" in explicit_rules:
        values["candidate_education_level"] = _education(
            explicit_rules["candidate_education_level"]
        )

    return JobSearchRules(**values)
