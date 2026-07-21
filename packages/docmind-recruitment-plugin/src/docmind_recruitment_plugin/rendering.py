from __future__ import annotations

from .extraction import ExtractionResult


_TITLE = "## JD 明确约束"
_NOTE = "> 仅整理原文明确内容；未列出的字段表示原文没有明确说明。"


def render_markdown(result: ExtractionResult) -> str:
    field_lines = [f"- {name}：{value}" for name, value in result.fields]
    return "\n".join((_TITLE, "", *field_lines, "", _NOTE))
