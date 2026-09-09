from __future__ import annotations

import re
from dataclasses import dataclass


# A normal retrieval prompt is bounded by at most 50 files, three 1,000-character
# chunks per file. This ceiling remains larger than that worst-case evidence text
# while preventing an unbounded response from reaching the terminal.
MAX_GENERATED_OUTPUT_CHARS = 262_144
PATHOLOGICAL_OUTPUT_MIN_CHARS = 4_096
MAX_REPEATED_CHARACTER_RUN = 4_096
MAX_PATHOLOGICAL_WHITESPACE_RATIO = 0.95

_SOURCE_ABSENCE_CLAIM_RE = re.compile(
    r"(?:现有|全部|原始|完整)?"
    r"(?P<source>(?:文件|资料|材料|文档|来源)(?:【[^】\n]+】)?)"
    r"(?:中|里)?"
    r"(?:并)?"
    r"(?:未(?:明确)?(?:提及|说明|规定|包含|找到|发现)"
    r"|没有(?:明确|相关)?(?:提及|说明|规定|内容|信息|记录)"
    r"|无(?:相关|明确)?(?:说明|规定|内容|信息|记录)"
    r"|不存在)"
)
_NEGATIVE_APPLICABILITY_RE = re.compile(
    r"(?:因此|所以|故而|由此可见)?"
    r"不适用于"
    r"(?P<scope>[^，。；！？\n]+)"
)
_EXPLICIT_BOUNDARY_EVIDENCE_RE = re.compile(
    r"(?:"
    r"(?:明确规定|明确写明|明确说明).{0,40}"
    r"(?:仅|只|不得|不适用|不包括|排除)"
    r"|(?:仅限|仅适用|只限|只适用|明确不适用|明确不包括)"
    r")"
)


@dataclass(frozen=True)
class GeneratedOutputValidation:
    text: str
    valid: bool
    reason: str | None
    raw_length: int
    stripped_length: int
    whitespace_ratio: float


def _has_pathological_character_run(text: str) -> bool:
    previous = None
    run_length = 0
    for character in text:
        if character == previous:
            run_length += 1
        else:
            previous = character
            run_length = 1
        if run_length >= MAX_REPEATED_CHARACTER_RUN:
            return True
    return False


def validate_generated_output(value: object) -> GeneratedOutputValidation:
    if not isinstance(value, str):
        return GeneratedOutputValidation(
            text="",
            valid=False,
            reason="non_text",
            raw_length=0,
            stripped_length=0,
            whitespace_ratio=0.0,
        )

    raw_length = len(value)
    text = value.strip()
    stripped_length = len(text)
    whitespace_count = sum(character.isspace() for character in value)
    whitespace_ratio = whitespace_count / raw_length if raw_length else 0.0

    reason = None
    if not text:
        reason = "empty_after_strip"
    elif raw_length > MAX_GENERATED_OUTPUT_CHARS:
        reason = "size_limit"
    elif (
        raw_length >= PATHOLOGICAL_OUTPUT_MIN_CHARS
        and whitespace_ratio >= MAX_PATHOLOGICAL_WHITESPACE_RATIO
    ):
        reason = "whitespace_dominance"
    elif (
        raw_length >= PATHOLOGICAL_OUTPUT_MIN_CHARS
        and _has_pathological_character_run(value)
    ):
        reason = "repeated_character_run"

    return GeneratedOutputValidation(
        text=text if reason is None else "",
        valid=reason is None,
        reason=reason,
        raw_length=raw_length,
        stripped_length=stripped_length,
        whitespace_ratio=whitespace_ratio,
    )


def enforce_bounded_absence_claims(text: str) -> str:
    """Keep limited retrieval gaps from becoming source-level absence facts."""
    if not isinstance(text, str) or not text:
        return text
    bounded = _SOURCE_ABSENCE_CLAIM_RE.sub(
        lambda match: (
            f"当前检索到的{match.group('source')}证据中暂未找到明确说明："
        ),
        text,
    )
    parts = re.split(r"([。！？\n])", bounded)
    for index in range(0, len(parts), 2):
        statement = parts[index]
        if not statement or _EXPLICIT_BOUNDARY_EVIDENCE_RE.search(statement):
            continue
        parts[index] = _NEGATIVE_APPLICABILITY_RE.sub(
            lambda match: (
                "当前检索到的证据不足以确认对"
                f"{match.group('scope')}不适用"
            ),
            statement,
        )
    return "".join(parts)


__all__ = [
    "GeneratedOutputValidation",
    "MAX_GENERATED_OUTPUT_CHARS",
    "MAX_PATHOLOGICAL_WHITESPACE_RATIO",
    "MAX_REPEATED_CHARACTER_RUN",
    "PATHOLOGICAL_OUTPUT_MIN_CHARS",
    "enforce_bounded_absence_claims",
    "validate_generated_output",
]
