from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from ai.generation_output import validate_generated_output
from ai.table_presentation import (
    MAX_TABLE_OUTPUT_TOKENS, StructuredTable, TableRenderOptions, build_keyed_table_response_schema,
    render_structured_table, validate_table_presentation,
)


_NO_SELECTION_MARKERS = (
    "暂无足够匹配", "没有足够匹配", "无足够匹配", "证据不足", "无法推荐",
    "不建议强行推荐", "没有可靠候选", "暂无合适", "没有合适",
)
_EMPTY_VALUE_MARKERS = {
    "", "无", "暂无", "没有", "未发现", "不存在",
    "暂无明显风险", "无明显风险", "没有明显风险", "未发现明显风险",
}
_EVIDENCE_OVERCLAIM_MARKERS = (
    "直接对应", "已经满足", "已满足", "完全满足", "完全符合", "充分证明",
    "可以证明", "表明你具备", "表明您具备", "表明用户具备",
)
_CANDIDATE_FIELD_PRIORITY = ("推荐对象", "最终推荐", "推荐岗位", "推荐方案")
_FIELD_ALIASES = {
    "推荐结论": "conclusion",
    "推荐对象": "candidate:推荐对象",
    "最终推荐": "candidate:最终推荐",
    "推荐岗位": "candidate:推荐岗位",
    "推荐方案": "candidate:推荐方案",
    "推荐理由": "reason",
    "方向匹配": "direction_matches",
    "主要匹配点": "direction_matches",
    "已有明确能力": "explicit_capabilities",
    "未被证据证明的要求": "unverified_requirements",
    "待确认要求": "unverified_requirements",
    "明显差距或风险": "risks",
    "差距或风险": "risks",
    "来源文件": "sources",
    "横向比较": "comparison",
    "差异与异常": "differences",
    "待确认信息": "missing_information",
    "推荐对象来源": "selected_sources",
    "下一步行动": "next_actions",
}
_UNRELIABLE_CANDIDATE_MARKERS = (
    "根据", "综合", "考虑", "因此", "所以", "以下", "最为匹配", "较为匹配",
    "推荐理由", "用户", "您", "你", "暂无", "证据", "候选",
)
_FILE_PATTERN = re.compile(
    r"([A-Za-z0-9_\-\u4e00-\u9fa5\\/:.\s]+?\."
    r"(?:txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx|png|jpg|jpeg|bmp|webp))\b",
    flags=re.IGNORECASE,
)

_COMPARISON_FIELDS = {
    "conclusion": "conclusion", "selected_candidate": "candidate:推荐对象",
    "reason": "reason", "comparison": "comparison", "differences": "differences",
    "missing_information": "missing_information", "next_actions": "next_actions",
    "selected_source_files": "selected_sources", "source_files": "sources",
}


def build_comparison_generation_config(generation_config, source_paths):
    """Constrain the existing decision fields, without introducing a domain schema."""
    properties = {
        key: ({"type": "array", "items": {"type": "string", "enum": list(source_paths)}}
              if key.endswith("source_files") else {"type": "string"})
        for key in _COMPARISON_FIELDS
    }
    properties["next_actions"] = {"type": "array", "items": {"type": "string"}}
    table_schema = build_keyed_table_response_schema()
    table_schema["nullable"] = True
    keyed_cells = table_schema["properties"]["rows"]["items"]
    table_schema["properties"]["rows"]["items"] = {
        "type": "object",
        "properties": {
            "cells": keyed_cells,
            "action_step": {"type": "integer", "nullable": True},
            "source_files": properties["source_files"],
        },
        "required": ["cells", "action_step", "source_files"],
    }
    properties["comparison_table"] = table_schema
    # Generate facts before deciding; the renderer still puts the conclusion
    # first for the user. Row references can point to the subsequent plan.
    properties = {
        key: properties[key] for key in (
            "comparison_table", "comparison", "differences", "missing_information",
            "next_actions", "conclusion", "reason", "selected_candidate",
            "selected_source_files", "source_files",
        )
    }
    token_limit = (
        generation_config.get("max_output_tokens") if isinstance(generation_config, Mapping)
        else getattr(generation_config, "max_output_tokens", None)
    )
    updates = {
        "response_mime_type": "application/json",
        "response_schema": {"type": "object", "properties": properties, "required": list(properties)},
        "max_output_tokens": min(token_limit or MAX_TABLE_OUTPUT_TOKENS, MAX_TABLE_OUTPUT_TOKENS),
    }
    thinking_config = (
        generation_config.get("thinking_config") if isinstance(generation_config, Mapping)
        else getattr(generation_config, "thinking_config", None)
    )
    if thinking_config is None:
        # Automatic thinking can consume the entire shared output budget before
        # the keyed rows finish. Reserve half the tokens for the validated payload;
        # preserve an explicitly configured thinking policy unchanged.
        thinking = {"thinking_budget": updates["max_output_tokens"] // 2}
        if isinstance(generation_config, Mapping):
            updates["thinking_config"] = thinking
        else:
            from google.genai.types import ThinkingConfig

            updates["thinking_config"] = ThinkingConfig(**thinking)
    if isinstance(generation_config, Mapping):
        return {**generation_config, **updates}
    return generation_config.model_copy(update=updates)


def _unique_comparison_fields(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("duplicate comparison field")
        payload[key] = value
    return payload


def _reject_comparison_constant(value):
    raise ValueError("non-standard JSON constant")


def _comparison_sections(
    answer_text: str, *, allow_partial: bool = False,
) -> dict[str, list[str]] | None:
    """JSON is authoritative when present; malformed objects must not become prose."""
    try:
        payload = json.loads(
            answer_text, object_pairs_hook=_unique_comparison_fields,
            parse_constant=_reject_comparison_constant,
        )
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not set(payload) <= set(_COMPARISON_FIELDS) | {"comparison_table"}:
        return None
    if not allow_partial and not set(_COMPARISON_FIELDS) <= set(payload):
        return None
    sections = {}
    for key, alias in _COMPARISON_FIELDS.items():
        if key not in payload:
            continue
        value = payload[key]
        if key.endswith("source_files") or (key == "next_actions" and isinstance(value, list)):
            if not isinstance(value, list) or not all(isinstance(path, str) for path in value):
                return None
            sections[alias] = value
        elif isinstance(value, str):
            sections[alias] = [value.strip()]
        else:
            return None
    if payload.get("comparison_table") is not None:
        sections["comparison_table"] = [json.dumps(payload["comparison_table"], ensure_ascii=False)]
    return sections


@dataclass(frozen=True)
class DecisionResult:
    conclusion: str = ""
    selected_candidate: str | None = None
    reason: str = ""
    direction_matches: str = ""
    explicit_capabilities: str = ""
    unverified_requirements: str = ""
    risks: str = ""
    source_files: tuple[str, ...] = ()
    comparison_requested: bool = False
    comparison: str = ""
    differences: str = ""
    missing_information: str = ""
    selected_source_files: tuple[str, ...] = ()
    next_actions: str = ""
    comparison_table: StructuredTable | None = None

    @property
    def has_selection(self) -> bool:
        return bool(self.selected_candidate and self.source_files)

    @property
    def explicit_user_facts(self) -> str:
        """Semantic alias retained alongside the legacy parser field name."""
        return self.explicit_capabilities


def _clean_value(text: str) -> str:
    value = re.sub(r"\s+", " ", (text or "").strip())
    return value.strip("*`#：:。；; ")


def _is_reliable_candidate(value: str) -> bool:
    candidate = _clean_value(value)
    if not candidate or len(candidate) > 80:
        return False
    if any(marker in candidate for marker in (*_NO_SELECTION_MARKERS, *_UNRELIABLE_CANDIDATE_MARKERS)):
        return False
    if re.search(r"[。！？!?；;]", candidate):
        return False
    return bool(re.search(r"[A-Za-z0-9\u4e00-\u9fa5]", candidate))


def _extract_sections(answer_text: str, *, preserve_preamble: bool = False) -> dict[str, list[str]]:
    labels = "|".join(sorted((re.escape(label) for label in _FIELD_ALIASES), key=len, reverse=True))
    header_re = re.compile(
        rf"^\s*(?:#{{1,6}}\s*)?(?:[-*]\s*)?(?:\*\*)?"
        rf"(?P<label>{labels})(?:\*\*)?\s*[:：]\s*(?:\*\*)?(?P<value>.*)$"
    )
    sections: dict[str, list[str]] = {}
    current_key: str | None = "comparison" if preserve_preamble else None
    for raw_line in (answer_text or "").splitlines():
        match = header_re.match(raw_line)
        if match:
            current_key = _FIELD_ALIASES[match.group("label")]
            sections.setdefault(current_key, [])
            value = _clean_value(match.group("value"))
            if value:
                sections[current_key].append(value)
            continue
        if current_key is not None:
            continuation = raw_line.strip()
            if continuation:
                sections.setdefault(current_key, []).append(continuation)
    return sections


def _section_text(sections: dict[str, list[str]], key: str) -> str:
    return "\n".join(sections.get(key, [])).strip()


def _extract_source_files(source_text: str) -> tuple[str, ...]:
    files: list[str] = []
    for match in _FILE_PATTERN.finditer(source_text or ""):
        candidate = re.sub(r"^[-*•\d.、)\s]+", "", match.group(1)).strip()
        candidate = re.sub(r"\s+", " ", candidate).strip("\"'[]【】（）() ，,。；;")
        if candidate and candidate not in files:
            files.append(candidate)
    return tuple(files)


def extract_explicit_user_facts(question: str) -> tuple[str, ...]:
    clauses = [
        _clean_value(part)
        for part in re.split(r"[，,。；;！？!?]+", question or "")
        if _clean_value(part)
    ]
    facts: list[str] = []
    for clause in clauses:
        if any(term in clause for term in ("推荐", "选择", "选一个", "选一份", "挑一个", "挑一份", "排序", "比较", "对比")):
            continue
        if clause not in facts:
            facts.append(clause)
    return tuple(facts)


def _normalize_visible_text(text: str) -> str:
    visible = re.sub(r"\s+", " ", (text or "").strip())
    visible = visible.replace("**", "")
    visible = re.sub(r"(?:^|\s)[*•]\s+", "；", visible)
    visible = re.sub(r"；\s*；+", "；", visible)
    visible = re.sub(r"[。；;]+\s*；", "；", visible)
    visible = re.sub(r"；\s*。", "。", visible)
    replacements = (
        ("现有用户信息", "你目前提供的信息"),
        ("用户未提供", "你目前还没有说明"),
        ("用户明确方向", "你明确提到的方向"),
        ("用户明确条件", "你明确提到的条件"),
        ("该对象", "它"),
        ("该岗位", "这个岗位"),
        ("该方案", "这个方案"),
        ("该项目", "这个项目"),
        ("您", "你"),
        ("用户", "你"),
    )
    for source, target in replacements:
        visible = visible.replace(source, target)
    visible = re.sub(
        r"你目前只有(.+?)(?=，而|，但|。|$)",
        r"目前能确认的是，你有\1",
        visible,
    )
    visible = re.sub(r"^(?:[-*•]|\d+[.、])\s*", "", visible)
    return visible.strip(" ，,。；;：: ")


def _fact_to_second_person(fact: str) -> str:
    text = _clean_value(fact)
    if not text:
        return ""

    work_match = re.fullmatch(
        r"(?:我|本人)?(?:当前|目前|现在)?(?:已经)?工作(?:了)?(?P<duration>.+)",
        text,
    )
    if work_match:
        duration = work_match.group("duration").strip()
        if duration:
            return f"你有{duration}工作经验"

    text = re.sub(r"^本人的", "你的", text)
    text = re.sub(r"^本人", "你", text)
    text = re.sub(r"^我的", "你的", text)
    text = re.sub(r"^我", "你", text)
    if not text.startswith(("你", "你的")) and re.match(
        r"^(?:主要|目前|当前|现在|平时|通常|一直|不接受|可以|不能|希望|偏好)",
        text,
    ):
        text = f"你{text}"
    return text


def render_explicit_user_facts(facts_text: str) -> str:
    raw_facts = [part for part in re.split(r"[；;\n]+", facts_text or "") if _clean_value(part)]
    visible_facts = [_fact_to_second_person(fact) for fact in raw_facts]
    visible_facts = [fact for fact in visible_facts if fact]
    if not visible_facts:
        return ""
    if len(visible_facts) > 1 and visible_facts[0].startswith("你"):
        for idx in range(1, len(visible_facts)):
            if visible_facts[idx].startswith("你主要"):
                visible_facts[idx] = visible_facts[idx][1:]
    return "，".join(visible_facts)


def _is_empty_value(text: str) -> bool:
    normalized = re.sub(r"[\s，,。；;：:！!？?]+", "", (text or "").strip())
    return normalized in _EMPTY_VALUE_MARKERS


def _as_sentence(text: str) -> str:
    value = _normalize_visible_text(text)
    if not value:
        return ""
    return value if value.endswith(("。", "！", "？")) else f"{value}。"


def _has_evidence_overclaim(text: str) -> bool:
    normalized = _normalize_visible_text(text)
    return any(marker in normalized for marker in _EVIDENCE_OVERCLAIM_MARKERS)


def _referenced_context_files(text: str, source_paths) -> tuple[str, ...]:
    # Match exact context identities; do not strip numeric filename prefixes as list numbers.
    return tuple(dict.fromkeys(
        path for path in source_paths
        if re.search(r"(?<![\w/\\.\-])" + re.escape(path) + r"(?![\w/\\.\-])", text)
    ))


def _comparison_table(sections, source_paths) -> StructuredTable | None:
    """Bind keyed facts and row actions before using the existing table wheel.

    The global plan owns action wording, conditions and order. Row actions
    contain only a 1-based reference to a complete step of that plan.
    Selection provenance is deliberately not used to infer action priority.
    """
    raw = _section_text(sections, "comparison_table")
    if not raw:
        return None
    payload = json.loads(raw)
    if not isinstance(payload, dict) or set(payload) != {"columns", "rows"}:
        return None
    columns, rows = payload["columns"], payload["rows"]
    if not isinstance(columns, list) or not isinstance(rows, list):
        return None
    actions = sections.get("next_actions", [])
    bound_rows = []
    facts = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"cells", "action_step", "source_files"}:
            return None
        cells, step, citations = row["cells"], row["action_step"], row["source_files"]
        if (not isinstance(cells, list) or not isinstance(citations, list) or not citations
                or any(not isinstance(path, str) or path not in source_paths for path in citations)):
            return None
        if step is not None and (type(step) is not int or not 1 <= step <= len(actions)):
            return None
        bound = []
        for cell in cells:
            if (not isinstance(cell, dict) or set(cell) != {"column", "value"}
                    or not isinstance(cell["column"], str) or not isinstance(cell["value"], str)):
                return None
            column, value = cell["column"], cell["value"]
            bound.append({"column": column, "value": value})
            facts.append(value)
        action = actions[step - 1] if step is not None else ""
        source = "\n".join(citations)
        bound.extend(({"column": "下一步", "value": action}, {"column": "来源", "value": source}))
        facts.extend((action, source))
        bound_rows.append(bound)
    validation = validate_table_presentation(
        {"columns": [*columns, "下一步", "来源"], "rows": bound_rows}, previous_answer="\n".join(facts),
    )
    return validation.table if validation.valid else None


def _table_evidence(table: StructuredTable | None) -> str:
    return "\n".join(cell for row in table.rows for cell in row) if table else ""


def parse_decision_result(
    answer_text: str, *, user_question: str = "", comparison_source_files=None,
) -> DecisionResult | None:
    if comparison_source_files:
        validation = validate_generated_output(answer_text)
        if not validation.valid or validation.text.startswith(("[", "```")):
            return None
    structured_comparison = bool(comparison_source_files) and (answer_text or "").lstrip().startswith("{")
    sections = (
        _comparison_sections(answer_text) if structured_comparison else
        _extract_sections(answer_text, preserve_preamble=bool(comparison_source_files))
    )
    if not sections:
        return None

    candidate = None
    for label in _CANDIDATE_FIELD_PRIORITY:
        value = _section_text(sections, f"candidate:{label}")
        if _is_reliable_candidate(value):
            candidate = _clean_value(value)
            break

    conclusion = _section_text(sections, "conclusion")
    if conclusion and any(marker in conclusion for marker in _NO_SELECTION_MARKERS):
        candidate = None

    if comparison_source_files:
        comparison = _section_text(sections, "comparison")
        table = _comparison_table(sections, comparison_source_files)
        if "comparison_table" in sections and table is None:
            return None
        differences = _section_text(sections, "differences") or _section_text(sections, "risks")
        missing = _section_text(sections, "missing_information") or _section_text(sections, "unverified_requirements")
        selected_sources = _referenced_context_files(
            _section_text(sections, "selected_sources"), comparison_source_files,
        )
        if not selected_sources or _is_empty_value(candidate or ""):
            candidate = None
        # Selection provenance is narrower than the evidence used by the answer.
        # Bind only sources actually cited in delivered sections, never the entire retrieval set.
        evidence_text = "\n".join((
            conclusion, _section_text(sections, "reason"), comparison, differences, _table_evidence(table),
            missing, _section_text(sections, "next_actions"), _section_text(sections, "sources"),
            _section_text(sections, "selected_sources") if candidate else "",
        ))
        sources = _referenced_context_files(evidence_text, comparison_source_files)
        if not sources:
            return None
        return DecisionResult(
            conclusion=conclusion,
            selected_candidate=candidate,
            reason=_section_text(sections, "reason"),
            source_files=sources,
            comparison_requested=True,
            comparison=comparison,
            differences=differences,
            missing_information=missing,
            selected_source_files=selected_sources if candidate else (),
            next_actions=_section_text(sections, "next_actions"),
            comparison_table=table,
        )

    explicit_facts = extract_explicit_user_facts(user_question)
    explicit_capabilities = "；".join(explicit_facts) if explicit_facts else _section_text(
        sections,
        "explicit_capabilities",
    )
    sources = _extract_source_files(_section_text(sections, "sources"))
    if candidate and not sources:
        candidate = None

    return DecisionResult(
        conclusion=conclusion,
        selected_candidate=candidate,
        reason=_section_text(sections, "reason"),
        direction_matches=_section_text(sections, "direction_matches"),
        explicit_capabilities=explicit_capabilities,
        unverified_requirements=_section_text(sections, "unverified_requirements"),
        risks=_section_text(sections, "risks"),
        source_files=sources if candidate else (),
    )


def build_comparison_prose_fallback(answer_text: str, source_paths) -> DecisionResult | None:
    """Deliver validated comparison facts without creating selection authority.

    A complete JSON object may omit schema fields; a broken object is never
    salvaged by extracting strings or repairing its syntax. Reuse the existing
    prose renderer and exact citation binding for the deliverable fields.
    """
    validation = validate_generated_output(answer_text)
    if not validation.valid:
        return None
    text = validation.text
    if text.startswith("```"):
        fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, flags=re.DOTALL)
        if not fenced:
            return None
        text = fenced.group(1).strip()
        if not text.startswith("{"):
            return None
    if text.startswith("{"):
        sections = _comparison_sections(text, allow_partial=True)
        if not sections:
            return None
        fields = {
            key: _section_text(sections, key)
            for key in ("conclusion", "reason", "comparison", "differences",
                        "missing_information", "next_actions")
        }
        table = _comparison_table(sections, source_paths)
        if table is None and not any(not _is_empty_value(value) for value in fields.values()):
            return None
        evidence = "\n".join((*fields.values(), _section_text(sections, "sources"), _table_evidence(table)))
        fields["comparison_table"] = table
    elif text.startswith("["):
        return None
    else:
        fields = {"comparison": text}
        evidence = text
    sources = _referenced_context_files(evidence, source_paths)
    if not sources:
        return None
    return DecisionResult(**fields, source_files=sources, comparison_requested=True)


def render_decision_result(result: DecisionResult) -> str:
    if result.comparison_requested:
        blocks = []
        if result.conclusion:
            blocks.append(f"建议：\n\n{result.conclusion}")
        elif result.has_selection:
            blocks.append(f"当前比较选定对象：**{result.selected_candidate}**（适用条件见下文）。")
        if result.reason:
            blocks.append(result.reason)
        comparison = result.comparison
        if result.comparison_table is not None and (
            len(result.comparison_table.rows) > 1 or _is_empty_value(comparison)
        ):
            comparison = render_structured_table(
                result.comparison_table, TableRenderOptions(missing_value="待确认"),
            )
        for label, value in (
            ("横向比较", comparison),
            ("差异与异常", result.differences),
            ("待确认信息", result.missing_information),
            ("下一步行动", result.next_actions),
        ):
            if not _is_empty_value(value):
                blocks.append(f"{label}：\n\n{value}")
        if result.source_files:
            blocks.append("来源：" + "、".join(result.source_files))
        return "\n\n".join(blocks) or "当前证据不足以完成比较。"

    if not result.has_selection:
        return "目前没有足够匹配的候选。"

    blocks = [f"我更推荐 **{result.selected_candidate}**。"]

    reason = _normalize_visible_text(result.reason)
    direction = _normalize_visible_text(result.direction_matches)
    has_unverified = not _is_empty_value(result.unverified_requirements)
    if reason and has_unverified and _has_evidence_overclaim(reason):
        reason = ""
    rationale = reason or direction
    if rationale:
        blocks.append(_as_sentence(f"主要是因为{rationale}"))
    else:
        blocks.append("主要是因为它与你明确提到的方向相对更接近。")

    boundary_parts = ["不过，方向匹配不代表已经满足全部要求。"]
    visible_facts = render_explicit_user_facts(result.explicit_user_facts)
    if visible_facts:
        boundary_parts.append(f"目前能确认的是，{visible_facts}。")
    if has_unverified:
        unverified = _normalize_visible_text(result.unverified_requirements)
        boundary_parts.append(
            f"还需要确认的是，{unverified}。这些从你目前提供的信息里还无法确认。"
        )
    blocks.append("".join(boundary_parts))

    if not _is_empty_value(result.risks):
        risk = _normalize_visible_text(result.risks)
        if risk:
            blocks.append(_as_sentence(f"另外需要留意：{risk}"))

    blocks.append("来源：" + "、".join(result.source_files))
    return "\n\n".join(blocks)


__all__ = [
    "DecisionResult",
    "build_comparison_prose_fallback",
    "build_comparison_generation_config",
    "extract_explicit_user_facts",
    "parse_decision_result",
    "render_explicit_user_facts",
    "render_decision_result",
]
