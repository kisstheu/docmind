from __future__ import annotations

from dataclasses import dataclass
import re

from .recognition import (
    FIELD_LABELS,
    LabelOccurrence,
    StructuralMatch,
    recognize_query,
    value_after,
)


FIELD_ORDER = (
    "岗位名称",
    "薪资",
    "工作地点",
    "工作制",
    "加班或大小周",
    "外包",
    "驻场",
    "学历",
    "经验",
    "技术要求",
)
_LIST_MARKER = re.compile(
    r"(?:^|\s)(?:\d{1,2}\.(?!\d)|\d{1,2}[、)]|[-*•])\s*"
)
_FRAGMENT_SPLIT = re.compile(r"(?:\r?\n)+|[；;。，,]+")
_LATIN_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9+#]*(?:[._/-][A-Za-z0-9+#]+)*"
)
_TECHNICAL_ACTION = (
    r"(?:必须|优先)?(?:熟悉|掌握|了解|精通|使用|运用|应用|开发|设计|部署|配置|调试|编写|维护|优化)"
)
_QUALIFIED_LATIN_TOKEN = re.compile(
    rf"{_TECHNICAL_ACTION}[^\r\n；;。]{{0,32}}{_LATIN_TOKEN.pattern}",
    re.IGNORECASE,
)
_QUALIFIED_CHINESE_PHRASE = re.compile(
    rf"{_TECHNICAL_ACTION}[^\r\n；;。]{{0,6}}[\u3400-\u9fff]{{2,}}"
)
_SOFT_SKILL_SIGNAL = re.compile(
    r"沟通|协作|合作|责任心|抗压|表达|协调|主动性|执行力|"
    r"学习能力|认真负责|communication|teamwork|collaboration|leadership",
    re.IGNORECASE,
)
_NON_TECHNICAL_ONLY = re.compile(
    r"^(?:学历(?:要求)?[:：]?)?(?:大专|本科|硕士|博士)(?:及|或)?以上$|"
    r"^(?:工作)?经验(?:要求)?[:：]?.*$|"
    r"^\d+\s*年(?:以上|以下)?(?:工作)?经验$|"
    r"^(?:可|能|需|要|接受).{0,8}(?:出差|加班|驻场|倒班|轮班).*$"
)
@dataclass(frozen=True)
class ExtractionResult:
    fields: tuple[tuple[str, str], ...]


def _deduplicate(values: list[str]) -> list[str]:
    unique: list[str] = []
    keys: set[str] = set()
    for value in values:
        cleaned = value.strip(" \t\r\n-—:：;；,，。")
        key = "".join(cleaned.casefold().split())
        if cleaned and key not in keys:
            keys.add(key)
            unique.append(cleaned)
    return unique


def _field_values(match: StructuralMatch, labels: tuple[str, ...]) -> list[str]:
    values = [
        value_after(match.text, occurrence, match.occurrences)
        for occurrence in match.occurrences
        if occurrence.label in labels
    ]
    return _deduplicate(values)


def _has_code_style_token(fragment: str) -> bool:
    for match in _LATIN_TOKEN.finditer(fragment):
        token = match.group()
        alphanumeric = "".join(character for character in token if character.isalnum())
        if any(character.isdigit() for character in token):
            return True
        if any(character in "+#._/" for character in token):
            return True
        if len(alphanumeric) >= 3 and alphanumeric.isupper():
            return True
        if any(character.isupper() for character in alphanumeric[1:]):
            return True
    return False


def _is_technical_fragment(fragment: str) -> bool:
    code_style = _has_code_style_token(fragment)
    qualified_latin = _QUALIFIED_LATIN_TOKEN.search(fragment) is not None
    if _SOFT_SKILL_SIGNAL.search(fragment) and not (code_style or qualified_latin):
        return False
    if _NON_TECHNICAL_ONLY.fullmatch(fragment):
        return False
    return (
        code_style
        or qualified_latin
        or _QUALIFIED_CHINESE_PHRASE.search(fragment) is not None
    )


def _technical_fragments(
    match: StructuralMatch,
    occurrence: LabelOccurrence,
) -> list[str]:
    content = value_after(match.text, occurrence, match.occurrences)
    fragments = _deduplicate(_FRAGMENT_SPLIT.split(_LIST_MARKER.sub("\n", content)))
    return [
        fragment
        for fragment in fragments
        if _is_technical_fragment(fragment)
    ]


def extract_constraints(query: str) -> ExtractionResult | None:
    match = recognize_query(query)
    if match is None:
        return None

    values: dict[str, str] = {"岗位名称": match.title_values[0]}
    valid_fields = {"岗位名称"}
    for field_name, labels in FIELD_LABELS.items():
        field_values = _field_values(match, labels)
        if not field_values:
            continue
        if len(field_values) == 1:
            values[field_name] = field_values[0]
            valid_fields.add(field_name)
        else:
            values[field_name] = "；".join(field_values) + "（原文存在冲突）"

    technical_values: list[str] = []
    for occurrence in match.requirement_occurrences:
        technical_values.extend(_technical_fragments(match, occurrence))
    technical_values = _deduplicate(technical_values)
    if technical_values:
        values["技术要求"] = "；".join(technical_values)
        valid_fields.add("技术要求")

    if "技术要求" not in valid_fields or len(valid_fields) < 4:
        return None
    return ExtractionResult(
        fields=tuple((field, values[field]) for field in FIELD_ORDER if field in values)
    )
