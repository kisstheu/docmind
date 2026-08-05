from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from decimal import Decimal
from types import MappingProxyType

import pytest

from docmind_recruitment_plugin import (
    INVALID_OPTIONS_CATEGORY,
    INVALID_OPTIONS_CODE,
    OPTIONS_SCHEMA_VERSION,
    PLUGIN_ID,
    EducationLevel,
    JobSearchRules,
    RecruitmentOptionsError,
    parse_recruitment_request_options,
)


RULE_FIELDS = (
    "minimum_monthly_salary_k",
    "require_double_weekends",
    "allow_outsourcing",
    "allow_onsite",
    "allowed_locations",
    "candidate_education_level",
    "candidate_relevant_years",
)


def _options(rules: object, *, version: object = OPTIONS_SCHEMA_VERSION):
    return {
        PLUGIN_ID: {
            "schema_version": version,
            "explicit_rules": rules,
        }
    }


def _assert_invalid(options) -> RecruitmentOptionsError:
    with pytest.raises(RecruitmentOptionsError) as caught:
        parse_recruitment_request_options(options)
    error = caught.value
    assert error.code == INVALID_OPTIONS_CODE
    assert error.category == INVALID_OPTIONS_CATEGORY
    assert str(error) == (
        "recruitment.options.invalid_payload: invalid_request_options"
    )
    assert repr(error) == (
        "RecruitmentOptionsError("
        "code='recruitment.options.invalid_payload', "
        "category='invalid_request_options')"
    )
    return error


def test_namespace_absent_returns_none_with_heterogeneous_namespaces() -> None:
    options = {
        "org.example.contract": {"mode": "strict"},
        "org.example.procurement": ["synthetic"],
        "org.example.project": True,
    }

    assert parse_recruitment_request_options(options) is None


def test_valid_empty_rules_returns_frozen_empty_job_search_rules() -> None:
    rules = parse_recruitment_request_options(_options({}))

    assert rules == JobSearchRules()
    with pytest.raises(FrozenInstanceError):
        rules.require_double_weekends = True  # type: ignore[misc,union-attr]


def test_all_seven_fields_convert_without_rewriting_values() -> None:
    options = _options(
        {
            "minimum_monthly_salary_k": 13.5,
            "require_double_weekends": True,
            "allow_outsourcing": False,
            "allow_onsite": False,
            "allowed_locations": [
                " 示例城市甲 ",
                "示例城市甲",
                "示例城市乙",
            ],
            "candidate_education_level": "associate",
            "candidate_relevant_years": 2.5,
        }
    )

    assert parse_recruitment_request_options(options) == JobSearchRules(
        minimum_monthly_salary_k=Decimal("13.5"),
        require_double_weekends=True,
        allow_outsourcing=False,
        allow_onsite=False,
        allowed_locations=(
            " 示例城市甲 ",
            "示例城市甲",
            "示例城市乙",
        ),
        candidate_education_level=EducationLevel.ASSOCIATE,
        candidate_relevant_years=Decimal("2.5"),
    )


@pytest.mark.parametrize(
    "namespace",
    (None, "rules", 1, 1.5, True, []),
)
def test_namespace_must_be_an_object(namespace: object) -> None:
    _assert_invalid({PLUGIN_ID: namespace})


def test_namespace_requires_exact_fields() -> None:
    for namespace in (
        {},
        {"schema_version": OPTIONS_SCHEMA_VERSION},
        {"explicit_rules": {}},
        {
            "schema_version": OPTIONS_SCHEMA_VERSION,
            "explicit_rules": {},
            "synthetic_unknown_namespace_key": "sentinel-value",
        },
    ):
        _assert_invalid({PLUGIN_ID: namespace})


@pytest.mark.parametrize(
    "version",
    (None, 1, 1.0, True, [], {}, "1.1", " 1.0", "1.0 "),
)
def test_schema_version_is_exact_required_string(version: object) -> None:
    _assert_invalid(_options({}, version=version))


@pytest.mark.parametrize(
    "rules",
    (None, "minimum salary 13", 1, 1.5, True, [], ["double weekends"]),
)
def test_explicit_rules_must_be_an_object(rules: object) -> None:
    _assert_invalid(_options(rules))


def test_unknown_rule_and_condition_object_are_rejected_atomically() -> None:
    for rules in (
        {"synthetic_unknown_rule": "sentinel-value"},
        {
            "minimum_monthly_salary_k": 13,
            "condition": {"when": "synthetic"},
        },
    ):
        _assert_invalid(_options(rules))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("minimum_monthly_salary_k", "13"),
        ("require_double_weekends", 1),
        ("allow_outsourcing", 0),
        ("allow_onsite", "false"),
        ("allowed_locations", "示例城市甲"),
        ("candidate_education_level", ["associate"]),
        ("candidate_relevant_years", "2"),
    ),
)
def test_each_rule_rejects_wrong_json_type(field: str, value: object) -> None:
    _assert_invalid(_options({field: value}))


@pytest.mark.parametrize("field", RULE_FIELDS)
def test_each_rule_rejects_explicit_null(field: str) -> None:
    _assert_invalid(_options({field: None}))


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("minimum_monthly_salary_k", True),
        ("minimum_monthly_salary_k", False),
        ("candidate_relevant_years", True),
        ("candidate_relevant_years", False),
        ("minimum_monthly_salary_k", -0.5),
        ("candidate_relevant_years", -1),
        ("minimum_monthly_salary_k", float("nan")),
        ("minimum_monthly_salary_k", float("inf")),
        ("minimum_monthly_salary_k", float("-inf")),
        ("candidate_relevant_years", float("nan")),
        ("candidate_relevant_years", float("inf")),
        ("candidate_relevant_years", float("-inf")),
    ),
)
def test_numbers_reject_bool_negative_and_nonfinite(
    field: str,
    value: object,
) -> None:
    _assert_invalid(_options({field: value}))


@pytest.mark.parametrize(
    ("field", "wire", "expected"),
    (
        ("minimum_monthly_salary_k", 13, Decimal("13")),
        ("minimum_monthly_salary_k", 13.25, Decimal("13.25")),
        ("candidate_relevant_years", 2, Decimal("2")),
        ("candidate_relevant_years", 2.5, Decimal("2.5")),
    ),
)
def test_numbers_use_decimal_string_conversion(
    field: str,
    wire: int | float,
    expected: Decimal,
) -> None:
    rules = parse_recruitment_request_options(_options({field: wire}))

    assert getattr(rules, field) == expected
    assert isinstance(getattr(rules, field), Decimal)


@pytest.mark.parametrize("value", ("", " ", "college", "BACHELOR", 1))
def test_education_rejects_values_outside_wire_enum(value: object) -> None:
    _assert_invalid(_options({"candidate_education_level": value}))


@pytest.mark.parametrize("level", tuple(EducationLevel))
def test_all_education_wire_values_map_to_enum(level: EducationLevel) -> None:
    rules = parse_recruitment_request_options(
        _options({"candidate_education_level": level.value})
    )

    assert rules.candidate_education_level is level


@pytest.mark.parametrize(
    "locations",
    ([1], [True], [None], [{}], [[]], [""], ["   "]),
)
def test_location_items_must_be_nonblank_strings(locations: list[object]) -> None:
    _assert_invalid(_options({"allowed_locations": locations}))


def test_locations_preserve_order_duplicates_and_empty_array_semantics() -> None:
    locations = ["示例城市乙", "示例城市甲", "示例城市乙"]
    rules = parse_recruitment_request_options(
        _options({"allowed_locations": locations})
    )
    empty = parse_recruitment_request_options(
        _options({"allowed_locations": []})
    )

    assert rules.allowed_locations == tuple(locations)
    assert empty.allowed_locations == ()


def test_missing_fields_keep_job_search_rules_unset_values() -> None:
    rules = parse_recruitment_request_options(
        _options({"require_double_weekends": False})
    )

    assert rules == JobSearchRules(require_double_weekends=False)


def test_success_and_failure_leave_all_input_containers_unchanged() -> None:
    successful = _options(
        {
            "minimum_monthly_salary_k": 13.25,
            "allowed_locations": [" 示例城市甲 ", "示例城市甲"],
        }
    )
    invalid = _options(
        {
            "require_double_weekends": True,
            "synthetic_unknown_rule": {"nested": ["sentinel-value"]},
        }
    )
    successful_before = deepcopy(successful)
    invalid_before = deepcopy(invalid)

    parse_recruitment_request_options(successful)
    _assert_invalid(invalid)

    assert successful == successful_before
    assert invalid == invalid_before


def test_read_only_mapping_is_supported_without_mutation() -> None:
    rules = MappingProxyType({"require_double_weekends": True})
    namespace = MappingProxyType(
        {
            "schema_version": OPTIONS_SCHEMA_VERSION,
            "explicit_rules": rules,
        }
    )
    options = MappingProxyType({PLUGIN_ID: namespace})

    assert parse_recruitment_request_options(options) == JobSearchRules(
        require_double_weekends=True
    )


def test_own_namespace_wins_among_three_heterogeneous_namespaces() -> None:
    options = {
        "org.example.contract": {"synthetic": "value"},
        PLUGIN_ID: {
            "schema_version": OPTIONS_SCHEMA_VERSION,
            "explicit_rules": {"allow_outsourcing": False},
        },
        "org.example.procurement": [1, 2],
        "org.example.project": None,
    }

    assert parse_recruitment_request_options(options) == JobSearchRules(
        allow_outsourcing=False
    )


def test_errors_and_output_are_fully_redacted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinels = (
        "synthetic_unknown_rule",
        "987654.25",
        "示例城市隐私哨兵",
        "doctorate-sentinel",
        "12345",
    )
    options = _options(
        {
            "synthetic_unknown_rule": {
                "salary": sentinels[1],
                "location": sentinels[2],
                "education": sentinels[3],
                "years": sentinels[4],
            }
        }
    )

    error = _assert_invalid(options)
    output = capsys.readouterr()
    exposed = str(error) + repr(error) + output.out + output.err

    assert all(sentinel not in exposed for sentinel in sentinels)
    assert output.out == ""
    assert output.err == ""
