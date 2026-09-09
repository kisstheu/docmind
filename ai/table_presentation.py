from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from io import StringIO
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text


MAX_TABLE_COLUMNS = 12
MAX_TABLE_ROWS = 200
MAX_TABLE_COLUMN_CHARS = 80
MAX_TABLE_CELL_CHARS = 2_048
MAX_TABLE_TOTAL_CHARS = 65_536
MAX_TABLE_OUTPUT_TOKENS = 8_192
MIN_RENDER_WIDTH = 40
MAX_RENDER_WIDTH = 240
MIN_COLUMN_WIDTH = 4
MAX_COLUMN_WIDTH = 80

_ColumnText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=MAX_TABLE_COLUMN_CHARS),
]
_CellText = Annotated[str, StringConstraints(max_length=MAX_TABLE_CELL_CHARS)]
_SchemaRow = Annotated[
    list[_CellText],
    Field(min_length=1, max_length=MAX_TABLE_COLUMNS),
]


class TablePresentationSchema(BaseModel):
    """Strict internal schema; local validation remains authoritative."""

    model_config = ConfigDict(extra="forbid", strict=True)

    columns: Annotated[
        list[_ColumnText],
        Field(min_length=1, max_length=MAX_TABLE_COLUMNS),
    ]
    rows: Annotated[
        list[_SchemaRow],
        Field(min_length=1, max_length=MAX_TABLE_ROWS),
    ]


@dataclass(frozen=True)
class StructuredTable:
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    row_identities: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TableRenderOptions:
    """Local-only display settings for an already validated table."""

    width: int = 120
    compact: bool = False
    wrap: bool = True
    balanced: bool = False
    column_widths: tuple[tuple[str, int], ...] = ()
    missing_value: str = ""


@dataclass(frozen=True)
class TablePresentationValidation:
    table: StructuredTable | None
    valid: bool
    reason: str | None


@dataclass(frozen=True)
class TableRefinement:
    table: StructuredTable | None
    options: TableRenderOptions | None
    valid: bool
    reason: str | None


def _build_google_table_response_schema() -> dict[str, object]:
    """Return the OpenAPI subset accepted by Google GenAI response_schema.

    Bounds and extra-field rejection intentionally stay in the strict local
    validator: the Gemini response_schema endpoint does not accept the
    ``additionalProperties`` emitted by the internal Pydantic model.
    """
    string_items = {"type": "string"}
    return {
        "type": "object",
        "properties": {
            "columns": {
                "type": "array",
                "items": dict(string_items),
            },
            "rows": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": dict(string_items),
                },
            },
        },
        "required": ["columns", "rows"],
    }


def build_table_generation_config(generation_config):
    """Add a provider-compatible native schema without mutating shared config."""
    current_max_tokens = (
        generation_config.get("max_output_tokens")
        if isinstance(generation_config, Mapping)
        else getattr(generation_config, "max_output_tokens", None)
    )
    max_output_tokens = (
        min(current_max_tokens, MAX_TABLE_OUTPUT_TOKENS)
        if isinstance(current_max_tokens, int) and current_max_tokens > 0
        else MAX_TABLE_OUTPUT_TOKENS
    )
    updates = {
        "response_mime_type": "application/json",
        "response_schema": _build_google_table_response_schema(),
        "max_output_tokens": max_output_tokens,
        "temperature": 0.0,
    }
    if hasattr(generation_config, "model_copy"):
        return generation_config.model_copy(update=updates)
    if isinstance(generation_config, Mapping):
        return {**generation_config, **updates}
    raise TypeError("unsupported generation config")


def build_keyed_table_response_schema() -> dict[str, object]:
    """Provider-compatible sparse cells for the same local table contract."""
    schema = _build_google_table_response_schema()
    schema["properties"]["rows"]["items"]["items"] = {
        "type": "object",
        "properties": {"column": {"type": "string"}, "value": {"type": "string"}},
        "required": ["column", "value"],
    }
    return schema


def _invalid(reason: str) -> TablePresentationValidation:
    return TablePresentationValidation(table=None, valid=False, reason=reason)


def _normalize_source_span(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def validate_table_presentation(
    value: object,
    *,
    previous_answer: str,
) -> TablePresentationValidation:
    """Validate presentation scaffolding and factual cell authority."""
    if isinstance(value, TablePresentationSchema):
        value = value.model_dump()
    if not isinstance(value, Mapping):
        return _invalid("non_object")
    if set(value) != {"columns", "rows"}:
        return _invalid("unexpected_fields")

    columns = value.get("columns")
    rows = value.get("rows")
    if not isinstance(columns, list):
        return _invalid("columns_not_array")
    if not columns:
        return _invalid("columns_empty")
    if len(columns) > MAX_TABLE_COLUMNS:
        return _invalid("columns_limit")
    if not isinstance(rows, list):
        return _invalid("rows_not_array")
    if not rows:
        return _invalid("rows_empty")
    if len(rows) > MAX_TABLE_ROWS:
        return _invalid("rows_limit")

    normalized_columns: list[str] = []
    total_chars = 0
    for column in columns:
        if not isinstance(column, str):
            return _invalid("column_type")
        normalized = column.strip()
        if not normalized:
            return _invalid("column_empty")
        if len(normalized) > MAX_TABLE_COLUMN_CHARS:
            return _invalid("column_size_limit")
        normalized_columns.append(normalized)
        total_chars += len(normalized)
    if len(set(normalized_columns)) != len(normalized_columns):
        return _invalid("duplicate_columns")

    normalized_source = _normalize_source_span(previous_answer)
    if not normalized_source:
        return _invalid("previous_answer_empty")

    # Columns are bounded presentation scaffolding: their labels may abstract
    # the meaning of existing data and therefore need not be verbatim source
    # spans. Rows and cells are the factual payload and remain source-bound.
    normalized_rows: list[tuple[str, ...]] = []
    has_content = False
    for row in rows:
        if not isinstance(row, list):
            return _invalid("row_not_array")
        # Keyed cells are sparse: identity, never position or value content,
        # determines the final column. Keep the positional contract unchanged.
        if row and isinstance(row[0], Mapping):
            keyed = {}
            for cell in row:
                if not isinstance(cell, Mapping) or set(cell) != {"column", "value"}:
                    return _invalid("keyed_cell_fields")
                column = cell["column"]
                if not isinstance(column, str) or column not in normalized_columns:
                    return _invalid("unknown_cell_column")
                if column in keyed:
                    return _invalid("duplicate_cell_column")
                keyed[column] = cell["value"]
            row = [keyed.get(column, "") for column in normalized_columns]
        if len(row) != len(normalized_columns):
            return _invalid("row_width")
        normalized_row: list[str] = []
        for cell in row:
            if not isinstance(cell, str):
                return _invalid("cell_type")
            normalized = cell.strip()
            if len(normalized) > MAX_TABLE_CELL_CHARS:
                return _invalid("cell_size_limit")
            total_chars += len(normalized)
            if total_chars > MAX_TABLE_TOTAL_CHARS:
                return _invalid("total_size_limit")
            if normalized:
                has_content = True
                if _normalize_source_span(normalized) not in normalized_source:
                    return _invalid("cell_not_in_previous_answer")
            normalized_row.append(normalized)
        normalized_rows.append(tuple(normalized_row))

    if not has_content:
        return _invalid("empty_content")

    try:
        TablePresentationSchema.model_validate(
            {
                "columns": normalized_columns,
                "rows": [list(row) for row in normalized_rows],
            }
        )
    except ValueError:
        return _invalid("schema_validation")

    return TablePresentationValidation(
        table=StructuredTable(
            columns=tuple(normalized_columns),
            rows=tuple(normalized_rows),
        ),
        valid=True,
        reason=None,
    )


def parse_table_presentation(
    raw_text: str,
    *,
    parsed: object = None,
    previous_answer: str,
) -> TablePresentationValidation:
    value = parsed
    if value is None:
        try:
            value = json.loads(raw_text)
        except (TypeError, ValueError):
            return _invalid("invalid_json")
    return validate_table_presentation(value, previous_answer=previous_answer)


def _normalize_refinement_text(text: str) -> str:
    return re.sub(r"[。！？!?：:\s]+", "", (text or "").strip().lower())


def _strip_refinement_softeners(text: str) -> str:
    normalized = _normalize_refinement_text(text)
    normalized = re.sub(
        r"^(?:(?:那|那么|那就)|(?:可以|能否|能不能|请|请你|麻烦|麻烦你|帮我|给我))*",
        "",
        normalized,
    )
    return re.sub(
        r"(?:一下|下|看看|看下|看一下)?(?:吧|吗|呢|呀|啊)?$",
        "",
        normalized,
    )


def _column_key(text: str) -> str:
    return re.sub(r"[，,、/\\|·\s]+", "", (text or "").strip().casefold())


def _column_nominal_base(text: str) -> str | None:
    """Return the referent behind a generic name-label suffix, if present."""
    for suffix in ("名称", "名字", "名"):
        if text.endswith(suffix) and len(text) > len(suffix):
            return text[: -len(suffix)]
    return None


def _resolve_column(table_data: StructuredTable, reference: str) -> int | None:
    token = _column_key(reference)
    exact_indices = [
        index
        for index, column in enumerate(table_data.columns)
        if token == _column_key(column)
    ]
    if len(exact_indices) == 1:
        return exact_indices[0]
    if len(exact_indices) > 1:
        return None

    token = re.sub(r"(?:这一)?列$", "", token)
    exact_indices = [
        index
        for index, column in enumerate(table_data.columns)
        if token == _column_key(column)
    ]
    if len(exact_indices) == 1:
        return exact_indices[0]
    if len(exact_indices) > 1:
        return None

    nominal_base = _column_nominal_base(token)
    if nominal_base is not None:
        nominal_indices = []
        for index, column in enumerate(table_data.columns):
            column_token = _column_key(column)
            column_base = _column_nominal_base(column_token) or column_token
            if column_token == nominal_base or column_base == nominal_base:
                nominal_indices.append(index)
        if len(nominal_indices) == 1:
            return nominal_indices[0]
        if len(nominal_indices) > 1:
            return None

    ordinal_match = re.fullmatch(r"第?([一二两三四五六七八九十\d]+)", token)
    if ordinal_match:
        ordinal_text = ordinal_match.group(1)
        chinese_ordinals = {
            "一": 1,
            "二": 2,
            "两": 2,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        ordinal = (
            int(ordinal_text)
            if ordinal_text.isdigit()
            else chinese_ordinals.get(ordinal_text)
        )
        if ordinal is not None and 1 <= ordinal <= len(table_data.columns):
            return ordinal - 1
    return None


def _natural_cell_key(value: str) -> tuple[tuple[int, object], ...]:
    parts = re.split(r"(-?\d+(?:\.\d+)?)", value.casefold())
    key: list[tuple[int, object]] = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"-?\d+(?:\.\d+)?", part):
            key.append((0, float(part)))
        else:
            key.append((1, part))
    return tuple(key)


def _reorder_columns(
    table_data: StructuredTable,
    ordered_indices: list[int],
) -> StructuredTable:
    remaining = [
        index for index in range(len(table_data.columns)) if index not in ordered_indices
    ]
    indices = ordered_indices + remaining
    return StructuredTable(
        columns=tuple(table_data.columns[index] for index in indices),
        rows=tuple(
            tuple(row[index] for index in indices)
            for row in table_data.rows
        ),
        row_identities=table_data.row_identities,
    )


def bind_structured_table_row_identities(
    table_data: StructuredTable,
    identity_aliases: Mapping[str, tuple[str, ...]],
) -> StructuredTable:
    """Attach identities only when every visible row has a unique exact binding."""
    canonical_identities = tuple(identity_aliases)
    existing = table_data.row_identities
    if (
        existing is not None
        and len(existing) == len(table_data.rows)
        and len(set(existing)) == len(existing)
        and set(existing) == set(canonical_identities)
    ):
        return table_data

    if (
        not canonical_identities
        or len(table_data.rows) != len(canonical_identities)
        or len(set(canonical_identities)) != len(canonical_identities)
    ):
        return replace(table_data, row_identities=None)

    resolved: list[str] = []
    for row in table_data.rows:
        matches = {
            identity
            for identity, aliases in identity_aliases.items()
            if any(cell == alias for cell in row for alias in aliases)
        }
        if len(matches) != 1:
            return replace(table_data, row_identities=None)
        resolved.append(matches.pop())

    if len(set(resolved)) != len(resolved) or set(resolved) != set(canonical_identities):
        return replace(table_data, row_identities=None)
    return replace(table_data, row_identities=tuple(resolved))


def _apply_single_refinement(
    table_data: StructuredTable,
    options: TableRenderOptions,
    clause: str,
) -> tuple[StructuredTable, TableRenderOptions] | None:
    q = _strip_refinement_softeners(clause)
    if not q:
        return None

    if re.fullmatch(r"(?:再)?(?:更)?(?:整齐|规整|清晰)(?:一点|一些|些|点)?", q):
        return table_data, replace(options, balanced=True, wrap=True)
    if re.fullmatch(r"(?:再)?(?:更)?紧凑(?:一点|一些|些|点)?", q):
        return table_data, replace(options, compact=True)
    if re.fullmatch(r"(?:表格|内容|单元格)?(?:自动)?换行(?:显示)?", q):
        return table_data, replace(options, wrap=True)
    if re.fullmatch(r"(?:表格|内容|单元格)?(?:不要|不再|取消)换行(?:显示)?", q):
        return table_data, replace(options, wrap=False)

    width_match = re.fullmatch(
        r"(?:(.+?))?列宽(?:调|改|设|设置|调整)?(?:成|为|到)?"
        r"(\d+)(?:个字符|字符)?",
        q,
    ) or re.fullmatch(
        r"(?:(.+?)(?:这一)?列)?宽度(?:调|改|设|设置|调整)?(?:成|为|到)?"
        r"(\d+)(?:个字符|字符)?",
        q,
    )
    if width_match:
        reference, raw_width = width_match.groups()
        width = int(raw_width)
        if not MIN_COLUMN_WIDTH <= width <= MAX_COLUMN_WIDTH:
            return None
        if reference:
            index = _resolve_column(table_data, reference)
            if index is None:
                return None
            widths = dict(options.column_widths)
            widths[table_data.columns[index]] = width
            return table_data, replace(
                options,
                column_widths=tuple(
                    (column, widths[column])
                    for column in table_data.columns
                    if column in widths
                ),
            )
        return table_data, replace(
            options,
            column_widths=tuple((column, width) for column in table_data.columns),
        )

    order_match = re.fullmatch(
        r"按(.+?)(?:的)?列顺序(?:排列|显示|调整|重排)?",
        q,
    )
    if order_match:
        references = [
            part
            for part in re.split(r"[，,、/\\|]+", order_match.group(1))
            if part
        ]
        indices = [_resolve_column(table_data, reference) for reference in references]
        if not indices or any(index is None for index in indices):
            return None
        resolved_indices = [int(index) for index in indices]
        if len(set(resolved_indices)) != len(resolved_indices):
            return None
        return _reorder_columns(table_data, resolved_indices), options

    position_match = re.fullmatch(
        r"把?(.+?)(?:这一)?列?(?:放|移|调整)(?:到)?(?:最)?(前面|后面|最后)",
        q,
    )
    if position_match:
        reference, position = position_match.groups()
        index = _resolve_column(table_data, reference)
        if index is None:
            return None
        other_indices = [
            current
            for current in range(len(table_data.columns))
            if current != index
        ]
        ordered_indices = (
            [index, *other_indices]
            if position == "前面"
            else [*other_indices, index]
        )
        return _reorder_columns(table_data, ordered_indices), options

    sort_match = re.fullmatch(
        r"按(.+?)(?:这一)?列?(?:(升序|降序|正序|倒序))?(?:排列|排序|排)",
        q,
    )
    if sort_match:
        reference, direction = sort_match.groups()
        index = _resolve_column(table_data, reference)
        if index is None:
            return None
        indexed_rows = sorted(
            enumerate(table_data.rows),
            key=lambda item: (
                not bool(item[1][index]),
                _natural_cell_key(item[1][index]),
            ),
            reverse=direction in {"降序", "倒序"},
        )
        identities = table_data.row_identities
        sorted_identities = (
            tuple(identities[row_index] for row_index, _row in indexed_rows)
            if identities is not None and len(identities) == len(table_data.rows)
            else None
        )
        return StructuredTable(
            columns=table_data.columns,
            rows=tuple(row for _row_index, row in indexed_rows),
            row_identities=sorted_identities,
        ), options

    return None


def refine_structured_table(
    table_data: StructuredTable | None,
    question: str,
    *,
    options: TableRenderOptions | None = None,
) -> TableRefinement:
    """Apply a closed set of presentation-only operations without changing cells."""
    if table_data is None:
        return TableRefinement(None, None, False, "table_missing")

    initial_options = options or TableRenderOptions()
    direct = _apply_single_refinement(table_data, initial_options, question)
    if direct is not None:
        refined_table, refined_options = direct
        return TableRefinement(refined_table, refined_options, True, None)

    clauses = [
        clause
        for clause in re.split(
            r"[，,；;]|(?:并且|而且|同时|另外|然后|还要|还得|且)",
            question or "",
        )
        if clause.strip()
    ]
    if not clauses:
        return TableRefinement(None, None, False, "request_empty")

    refined_table = table_data
    refined_options = initial_options
    for clause in clauses:
        applied = _apply_single_refinement(refined_table, refined_options, clause)
        if applied is None:
            return TableRefinement(None, None, False, "unsupported_operation")
        refined_table, refined_options = applied

    return TableRefinement(refined_table, refined_options, True, None)


def render_structured_table(
    table_data: StructuredTable,
    options: TableRenderOptions | None = None,
) -> str:
    """Render validated data as deterministic, markup-free terminal text."""
    render_options = options or TableRenderOptions()
    console_width = min(MAX_RENDER_WIDTH, max(MIN_RENDER_WIDTH, render_options.width))
    explicit_widths = dict(render_options.column_widths)
    balanced_max_width = max(
        MIN_COLUMN_WIDTH,
        (console_width - (2 * len(table_data.columns))) // len(table_data.columns),
    )
    table = Table(
        box=box.SIMPLE_HEAD,
        collapse_padding=True,
        padding=(0, 0 if render_options.compact else 1),
        show_edge=False,
        header_style=None,
    )
    for column in table_data.columns:
        column_width = explicit_widths.get(column)
        column_options: dict[str, object] = {
            "overflow": "fold" if render_options.wrap else "ellipsis",
            "no_wrap": not render_options.wrap,
        }
        if column_width is not None:
            column_options["width"] = column_width
        elif render_options.balanced:
            column_options["max_width"] = balanced_max_width
        table.add_column(Text(column), **column_options)
    for row in table_data.rows:
        table.add_row(*(Text(cell or render_options.missing_value) for cell in row))

    output = StringIO()
    Console(
        file=output,
        force_terminal=False,
        color_system=None,
        highlight=False,
        width=console_width,
    ).print(table)
    return output.getvalue().rstrip()


__all__ = [
    "MAX_TABLE_CELL_CHARS",
    "MAX_TABLE_COLUMNS",
    "MAX_TABLE_COLUMN_CHARS",
    "MAX_TABLE_OUTPUT_TOKENS",
    "MAX_TABLE_ROWS",
    "MAX_TABLE_TOTAL_CHARS",
    "StructuredTable",
    "TableRefinement",
    "TableRenderOptions",
    "TablePresentationSchema",
    "TablePresentationValidation",
    "bind_structured_table_row_identities",
    "build_table_generation_config",
    "build_keyed_table_response_schema",
    "parse_table_presentation",
    "refine_structured_table",
    "render_structured_table",
    "validate_table_presentation",
]
