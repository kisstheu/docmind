from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
import re

from .extraction import ExtractionResult, extract_constraints


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


_CONFLICT_MARKER = "原文存在冲突"
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
_EXPERIENCE_RANGE = re.compile(
    r"(?P<minimum>\d+(?:\.\d+)?)\s*"
    r"[-–—~～至到]\s*"
    r"(?P<maximum>\d+(?:\.\d+)?)\s*年"
)
_EXPERIENCE_MINIMUM = re.compile(
    r"(?:至少|不低于)?\s*(?P<minimum>\d+(?:\.\d+)?)\s*年(?:及?以上)?"
)


def _validate_decimal(name: str, value: Decimal | None) -> None:
    if value is None:
        return
    if not isinstance(value, Decimal):
        raise TypeError(f"{name} must be Decimal or None")
    if not value.is_finite() or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


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
