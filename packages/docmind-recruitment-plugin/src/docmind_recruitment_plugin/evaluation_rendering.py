from __future__ import annotations

from .extraction import ExtractionResult


_TITLE = "## JD 本身分析"
_BOUNDARY = (
    "> 当前没有加载明确求职规则；以下仅分析 JD 明示内容，"
    "不进行个性化匹配，也不作投递建议。"
)
_THRESHOLD_FIELDS = ("技术要求", "经验", "学历")
_CONDITION_FIELDS = (
    "薪资",
    "工作地点",
    "工作制",
    "加班或大小周",
    "外包",
    "驻场",
)


def _render_section(
    heading: str,
    fields: dict[str, str],
    field_order: tuple[str, ...],
) -> list[str]:
    lines = ["", f"### {heading}"]
    present = [(name, fields[name]) for name in field_order if name in fields]
    if not present:
        lines.append("- JD 未明确说明。")
        return lines
    lines.extend(f"- {name}：{value}" for name, value in present)
    return lines


def render_unpersonalized_evaluation(result: ExtractionResult) -> str:
    fields = dict(result.fields)
    lines = [_TITLE, "", _BOUNDARY, "", "### 岗位概况"]
    lines.append(f"- 岗位名称：{fields['岗位名称']}")
    lines.extend(_render_section("主要门槛", fields, _THRESHOLD_FIELDS))
    lines.extend(_render_section("明示条件", fields, _CONDITION_FIELDS))

    missing = [
        name
        for name in (*_THRESHOLD_FIELDS, *_CONDITION_FIELDS)
        if name not in fields
    ]
    lines.extend(("", "### 仍需确认"))
    if missing:
        lines.append(f"- JD 未明确说明：{'、'.join(missing)}。")
    else:
        lines.append("- 上述字段均有明示内容；其他未写事项仍需另行确认。")
    return "\n".join(lines)


__all__ = ["render_unpersonalized_evaluation"]
