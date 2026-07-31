from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, fields
from decimal import Decimal
import inspect
from pathlib import Path

import pytest

from docmind_domain_sdk import DomainRequest
from docmind_recruitment_plugin import (
    ConstraintAssessment,
    ConstraintStatus,
    EducationLevel,
    ExtractionResult,
    JobRuleComparison,
    JobSearchRules,
    RecruitmentJDPlugin,
    compare_job_search_rules,
    extract_and_compare,
    extract_constraints,
    render_job_rule_comparison,
)
from docmind_recruitment_plugin.comparison import (
    ConstraintAssessment as ComparisonConstraintAssessment,
)
from docmind_recruitment_plugin.comparison import (
    ConstraintStatus as ComparisonConstraintStatus,
)
from docmind_recruitment_plugin.comparison import (
    EducationLevel as ComparisonEducationLevel,
)
from docmind_recruitment_plugin.comparison import (
    JobRuleComparison as ComparisonJobRuleComparison,
)
from docmind_recruitment_plugin.comparison import (
    JobSearchRules as ComparisonJobSearchRules,
)
from docmind_recruitment_plugin.comparison import (
    compare_job_search_rules as comparison_compare_job_search_rules,
)
from docmind_recruitment_plugin.comparison import (
    extract_and_compare as comparison_extract_and_compare,
)


def _jd(
    *,
    salary: str | None = "16K–22K",
    work_schedule: str | None = "双休",
    outsourcing: str | None = None,
    onsite: str | None = None,
    location: str | None = "示例城市甲",
    education: str | None = "大专及以上",
    experience: str | None = "2年以上",
    technical: str = "熟悉 Python；掌握 SQL；了解 FastAPI",
) -> str:
    lines = [
        "职位名称：合成服务开发工程师",
        (
            "岗位职责：负责完全合成的服务接口、稳定性验证、交付记录和技术文档维护，"
            "按照明确流程完成测试与变更复核。"
        ),
        f"任职要求：{technical}",
    ]
    optional_fields = (
        ("薪资", salary),
        ("工作地点", location),
        ("工作制", work_schedule),
        ("是否外包", outsourcing),
        ("是否驻场", onsite),
        ("学历要求", education),
        ("经验要求", experience),
    )
    lines.extend(
        f"{label}：{value}"
        for label, value in optional_fields
        if value is not None
    )
    return "\n".join(lines)


def _comparison(query: str, **rule_values) -> JobRuleComparison:
    result = extract_and_compare(query, JobSearchRules(**rule_values))
    assert result is not None
    return result


def _assessment(
    result: JobRuleComparison,
    field: str,
):
    return next(item for item in result.assessments if item.field == field)


def _execute(query: str):
    request = DomainRequest(
        request_id="comparison-compatibility",
        query=query,
        source_scope=(),
    )
    return asyncio.run(RecruitmentJDPlugin().execute(request))


def test_public_comparison_facade_preserves_contract_exports() -> None:
    assert ComparisonEducationLevel is EducationLevel
    assert ComparisonConstraintStatus is ConstraintStatus
    assert ComparisonJobSearchRules is JobSearchRules
    assert ComparisonConstraintAssessment is ConstraintAssessment
    assert ComparisonJobRuleComparison is JobRuleComparison
    assert comparison_compare_job_search_rules is compare_job_search_rules
    assert comparison_extract_and_compare is extract_and_compare
    assert tuple(item.value for item in EducationLevel) == (
        "unrestricted",
        "secondary",
        "associate",
        "bachelor",
        "master",
        "doctorate",
    )
    assert tuple(item.value for item in ConstraintStatus) == (
        "match",
        "conflict",
        "unknown",
    )
    assert tuple(field.name for field in fields(JobSearchRules)) == (
        "minimum_monthly_salary_k",
        "require_double_weekends",
        "allow_outsourcing",
        "allow_onsite",
        "allowed_locations",
        "candidate_education_level",
        "candidate_relevant_years",
    )
    assert tuple(field.name for field in fields(ConstraintAssessment)) == (
        "field",
        "status",
        "jd_value",
        "rule_value",
        "jd_evidence",
        "reason",
        "confirmation_question",
    )
    assert tuple(field.name for field in fields(JobRuleComparison)) == (
        "assessments",
        "unassessed_technical_evidence",
    )


def test_explicit_rule_models_are_public_frozen_and_have_no_profile_defaults() -> None:
    rules = JobSearchRules()

    assert rules.minimum_monthly_salary_k is None
    assert rules.require_double_weekends is None
    assert rules.allow_outsourcing is None
    assert rules.allow_onsite is None
    assert rules.allowed_locations == ()
    assert rules.candidate_education_level is None
    assert rules.candidate_relevant_years is None
    with pytest.raises(FrozenInstanceError):
        rules.require_double_weekends = True  # type: ignore[misc]


@pytest.mark.parametrize(
    ("values", "error"),
    (
        ({"minimum_monthly_salary_k": 12.5}, TypeError),
        ({"candidate_relevant_years": 2.0}, TypeError),
        ({"minimum_monthly_salary_k": Decimal("-1")}, ValueError),
        ({"candidate_relevant_years": Decimal("NaN")}, ValueError),
        ({"allowed_locations": ["示例城市甲"]}, TypeError),
        ({"allowed_locations": ("",)}, ValueError),
    ),
)
def test_rules_reject_float_mutable_or_invalid_values(values, error) -> None:
    with pytest.raises(error):
        JobSearchRules(**values)


@pytest.mark.parametrize(
    ("salary", "minimum", "status"),
    (
        ("16K–22K", Decimal("14"), ConstraintStatus.MATCH),
        ("8K–11K", Decimal("14"), ConstraintStatus.CONFLICT),
        ("11K–18K", Decimal("14"), ConstraintStatus.UNKNOWN),
        ("薪资面议", Decimal("14"), ConstraintStatus.UNKNOWN),
        (None, Decimal("14"), ConstraintStatus.UNKNOWN),
    ),
)
def test_salary_comparison_uses_real_extraction_chain(
    salary: str | None,
    minimum: Decimal,
    status: ConstraintStatus,
) -> None:
    result = _comparison(
        _jd(salary=salary),
        minimum_monthly_salary_k=minimum,
    )
    assessment = _assessment(result, "salary")

    assert assessment.status is status
    assert assessment.rule_value == minimum
    assert assessment.jd_evidence == (() if salary is None else (salary,))
    if status is ConstraintStatus.UNKNOWN:
        assert assessment.confirmation_question is not None


@pytest.mark.parametrize(
    ("work_schedule", "status"),
    (
        ("周末双休", ConstraintStatus.MATCH),
        ("单休", ConstraintStatus.CONFLICT),
        ("大小周", ConstraintStatus.CONFLICT),
        (None, ConstraintStatus.UNKNOWN),
    ),
)
def test_work_schedule_comparison_uses_explicit_schedule_only(
    work_schedule: str | None,
    status: ConstraintStatus,
) -> None:
    result = _comparison(
        _jd(work_schedule=work_schedule),
        require_double_weekends=True,
    )
    assessment = _assessment(result, "work_schedule")

    assert assessment.status is status
    assert assessment.jd_evidence == (
        () if work_schedule is None else (work_schedule,)
    )


@pytest.mark.parametrize(
    ("outsourcing", "status"),
    (
        ("第三方签约", ConstraintStatus.CONFLICT),
        ("派遣", ConstraintStatus.CONFLICT),
        ("非外包，与用人主体直接签约", ConstraintStatus.MATCH),
        (None, ConstraintStatus.UNKNOWN),
        ("项目制", ConstraintStatus.UNKNOWN),
    ),
)
def test_outsourcing_comparison_does_not_guess_from_project_wording(
    outsourcing: str | None,
    status: ConstraintStatus,
) -> None:
    result = _comparison(
        _jd(outsourcing=outsourcing),
        allow_outsourcing=False,
    )

    assert _assessment(result, "outsourcing").status is status


@pytest.mark.parametrize(
    ("onsite", "status"),
    (
        ("长期驻客户现场办公", ConstraintStatus.CONFLICT),
        ("不驻场，本公司办公", ConstraintStatus.MATCH),
        ("需要现场沟通", ConstraintStatus.UNKNOWN),
        ("偶尔去客户处", ConstraintStatus.UNKNOWN),
        (None, ConstraintStatus.UNKNOWN),
    ),
)
def test_onsite_comparison_requires_long_term_onsite_evidence(
    onsite: str | None,
    status: ConstraintStatus,
) -> None:
    result = _comparison(_jd(onsite=onsite), allow_onsite=False)

    assert _assessment(result, "onsite").status is status


@pytest.mark.parametrize(
    ("location", "allowed", "status"),
    (
        (
            "示例城市甲",
            ("示例城市甲",),
            ConstraintStatus.MATCH,
        ),
        (
            "示例城市乙",
            ("示例城市甲",),
            ConstraintStatus.CONFLICT,
        ),
        (
            "示例城市甲、示例城市乙",
            ("示例城市甲",),
            ConstraintStatus.UNKNOWN,
        ),
        (
            "示例城市甲或示例城市乙",
            ("示例城市甲",),
            ConstraintStatus.UNKNOWN,
        ),
        (
            "工作地点待定",
            ("示例城市甲",),
            ConstraintStatus.UNKNOWN,
        ),
        (
            None,
            ("示例城市甲",),
            ConstraintStatus.UNKNOWN,
        ),
    ),
)
def test_location_comparison_uses_only_explicit_allowed_values(
    location: str | None,
    allowed: tuple[str, ...],
    status: ConstraintStatus,
) -> None:
    result = _comparison(_jd(location=location), allowed_locations=allowed)

    assert _assessment(result, "location").status is status


def test_location_applies_only_basic_suffix_and_whitespace_normalization() -> None:
    result = _comparison(
        _jd(location=" 示例城市甲市 "),
        allowed_locations=("示例城市甲",),
    )

    assert _assessment(result, "location").status is ConstraintStatus.MATCH


@pytest.mark.parametrize(
    ("education", "candidate", "status"),
    (
        (
            "大专及以上",
            EducationLevel.ASSOCIATE,
            ConstraintStatus.MATCH,
        ),
        (
            "本科及以上",
            EducationLevel.ASSOCIATE,
            ConstraintStatus.CONFLICT,
        ),
        (
            "本科优先",
            EducationLevel.ASSOCIATE,
            ConstraintStatus.UNKNOWN,
        ),
        (
            "统招本科及以上",
            EducationLevel.BACHELOR,
            ConstraintStatus.UNKNOWN,
        ),
        (
            None,
            EducationLevel.BACHELOR,
            ConstraintStatus.UNKNOWN,
        ),
    ),
)
def test_education_compares_only_unqualified_hard_minimum(
    education: str | None,
    candidate: EducationLevel,
    status: ConstraintStatus,
) -> None:
    result = _comparison(
        _jd(education=education),
        candidate_education_level=candidate,
    )
    assessment = _assessment(result, "education")

    assert assessment.status is status
    assert assessment.jd_evidence == (
        () if education is None else (education,)
    )


@pytest.mark.parametrize(
    ("experience", "candidate", "status"),
    (
        ("2年以上", Decimal("3"), ConstraintStatus.MATCH),
        ("3年以上", Decimal("2"), ConstraintStatus.CONFLICT),
        ("有经验者优先", Decimal("2"), ConstraintStatus.UNKNOWN),
        ("1～3年", Decimal("5"), ConstraintStatus.MATCH),
        (None, Decimal("2"), ConstraintStatus.UNKNOWN),
    ),
)
def test_experience_compares_hard_minimum_and_ignores_upper_bound(
    experience: str | None,
    candidate: Decimal,
    status: ConstraintStatus,
) -> None:
    result = _comparison(
        _jd(experience=experience),
        candidate_relevant_years=candidate,
    )

    assert _assessment(result, "experience").status is status


def test_one_jd_can_produce_match_conflict_and_unknown_with_evidence() -> None:
    result = _comparison(
        _jd(
            salary="17K–23K",
            work_schedule="大小周",
            onsite="需要现场沟通",
        ),
        minimum_monthly_salary_k=Decimal("14"),
        require_double_weekends=True,
        allow_onsite=False,
    )

    assert result.match_count == 1
    assert result.conflict_count == 1
    assert result.unknown_count == 1
    assert tuple(item.field for item in result.assessments) == (
        "salary",
        "work_schedule",
        "onsite",
    )
    assert all(item.jd_evidence for item in result.assessments)
    assert _assessment(result, "onsite").confirmation_question is not None


def test_technical_requirements_remain_unassessed_original_evidence() -> None:
    result = _comparison(
        _jd(technical="熟悉 Python；掌握 SQL；了解 FastAPI"),
        minimum_monthly_salary_k=Decimal("14"),
    )

    assert result.unassessed_technical_evidence == (
        "熟悉 Python",
        "掌握 SQL",
        "了解 FastAPI",
    )
    assert all(item.field != "technical" for item in result.assessments)


def test_existing_extraction_result_can_be_passed_directly_to_public_service() -> None:
    extracted = extract_constraints(_jd())
    assert extracted is not None
    assert isinstance(extracted, ExtractionResult)

    result = compare_job_search_rules(
        extracted,
        JobSearchRules(candidate_relevant_years=Decimal("4")),
    )

    assert _assessment(result, "experience").status is ConstraintStatus.MATCH


def test_rendering_has_fixed_sections_status_groups_evidence_and_questions() -> None:
    result = _comparison(
        _jd(
            salary="17K–23K",
            work_schedule="单休",
            outsourcing=None,
        ),
        minimum_monthly_salary_k=Decimal("14"),
        require_double_weekends=True,
        allow_outsourcing=False,
    )

    rendered = render_job_rule_comparison(result)

    assert rendered.index("### 明确符合") < rendered.index("### 明确冲突")
    assert rendered.index("### 明确冲突") < rendered.index(
        "### 信息缺失或需要确认"
    )
    assert rendered.index("### 信息缺失或需要确认") < rendered.index(
        "### 技术要求原文"
    )
    assert "JD证据：17K–23K" in rendered
    assert "JD证据：JD未明确" in rendered
    assert "建议确认：该岗位是否为外包、派遣或第三方签约？" in rendered
    assert "ConstraintStatus" not in rendered
    assert "建议投递" not in rendered
    assert "不建议投递" not in rendered
    assert "匹配百分比" not in rendered
    assert rendered == render_job_rule_comparison(result)


def test_assessment_order_is_stable_for_all_seven_fields() -> None:
    result = _comparison(
        _jd(outsourcing="否", onsite="不驻场"),
        minimum_monthly_salary_k=Decimal("14"),
        require_double_weekends=True,
        allow_outsourcing=False,
        allow_onsite=False,
        allowed_locations=("示例城市甲",),
        candidate_education_level=EducationLevel.ASSOCIATE,
        candidate_relevant_years=Decimal("3"),
    )

    assert tuple(item.field for item in result.assessments) == (
        "salary",
        "work_schedule",
        "outsourcing",
        "onsite",
        "location",
        "education",
        "experience",
    )


def test_missing_fields_are_unknown_and_never_raise() -> None:
    query = _jd(
        salary=None,
        work_schedule="弹性工作",
        outsourcing=None,
        onsite=None,
        location=None,
        education=None,
        experience="2年以上",
    )
    result = _comparison(
        query,
        minimum_monthly_salary_k=Decimal("14"),
        require_double_weekends=True,
        allow_outsourcing=False,
        allow_onsite=False,
        allowed_locations=("示例城市甲",),
        candidate_education_level=EducationLevel.ASSOCIATE,
    )

    assert all(
        item.status is ConstraintStatus.UNKNOWN
        for item in result.assessments
    )


def test_no_rules_leave_existing_plugin_1_0_output_unchanged() -> None:
    query = _jd()
    before = _execute(query)
    empty_comparison = extract_and_compare(query, JobSearchRules())
    after = _execute(query)

    assert empty_comparison is not None
    assert empty_comparison.assessments == ()
    assert before == after
    assert before.status == "handled"
    assert before.answer_markdown.startswith("## JD 明确约束\n\n")
    assert "单 JD 显式规则比较" not in before.answer_markdown


@pytest.mark.parametrize(
    "query",
    (
        "这是一段完全合成的普通项目说明，不是招聘 JD。",
        (
            "合同条款：履约地点为示例地点甲，技术要求为提交接口文档，"
            "经验参数只用于说明交付负责人资历。"
        ),
        (
            "采购规格：交付地点待定，供应商需要熟悉测试流程，"
            "学历与薪资字段不属于本采购文件的评价依据。"
        ),
    ),
)
def test_non_jd_and_cross_domain_text_do_not_enter_comparison(query: str) -> None:
    assert extract_and_compare(
        query,
        JobSearchRules(require_double_weekends=True),
    ) is None
    assert _execute(query).status == "abstain"


def test_multiple_complete_jd_blocks_keep_existing_abstain_boundary() -> None:
    first = _jd()
    second = _jd().replace(
        "职位名称：合成服务开发工程师",
        "职位名称：合成数据开发工程师",
        1,
    )
    query = first + "\n\n" + second

    assert extract_and_compare(
        query,
        JobSearchRules(minimum_monthly_salary_k=Decimal("14")),
    ) is None
    assert _execute(query).status == "abstain"


def test_comparison_source_has_no_personal_profile_or_external_dependency() -> None:
    source_path = (
        Path(inspect.getfile(JobSearchRules)).resolve().parent
        / "comparison.py"
    )
    source = source_path.read_text(encoding="utf-8")

    assert "WorkState" not in source
    assert "resume" not in source.casefold()
    assert "getenv" not in source
    assert "environ" not in source
    assert "requests" not in source
    assert "httpx" not in source
    assert "openai" not in source.casefold()
    assert "embedding" not in source.casefold()
