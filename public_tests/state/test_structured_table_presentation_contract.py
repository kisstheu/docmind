from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from google import genai
from google.genai import types

from ai.table_presentation import (
    MAX_TABLE_CELL_CHARS,
    MAX_TABLE_COLUMNS,
    MAX_TABLE_OUTPUT_TOKENS,
    MAX_TABLE_ROWS,
    StructuredTable,
    TablePresentationSchema,
    TableRenderOptions,
    build_table_generation_config,
    parse_table_presentation,
    refine_structured_table,
    render_structured_table,
    validate_table_presentation,
)
from app.dialog.task_semantics import is_table_presentation_request


def test_valid_multiline_chinese_table_is_locally_rendered_stably():
    previous_answer = "合成项甲负责接口，合成项乙负责数据处理。"
    validation = validate_table_presentation(
        {
            "columns": ["对象", "说明"],
            "rows": [["合成项甲", "接口"], ["合成项乙", "数据处理"]],
        },
        previous_answer=previous_answer,
    )

    assert validation == validation.__class__(
        table=StructuredTable(
            columns=("对象", "说明"),
            rows=(("合成项甲", "接口"), ("合成项乙", "数据处理")),
        ),
        valid=True,
        reason=None,
    )
    assert render_structured_table(validation.table) == (
        " 对象      说明     \n"
        "────────────────────\n"
        " 合成项甲  接口     \n"
        " 合成项乙  数据处理"
    )


@pytest.mark.parametrize(
    ("columns", "rows", "question", "expected_columns", "expected_rows"),
    [
        (
            ("候选人", "年限"),
            (("候选人X", "2"), ("候选人Y", "10")),
            "按年限降序排序",
            ("候选人", "年限"),
            (("候选人Y", "10"), ("候选人X", "2")),
        ),
        (
            ("条款", "期限"),
            (("条款甲", "30天"), ("条款乙", "7天")),
            "把期限列放到最前面",
            ("期限", "条款"),
            (("30天", "条款甲"), ("7天", "条款乙")),
        ),
        (
            ("采购项", "包装"),
            (("采购项甲", "密封"),),
            "采购项列宽调到16",
            ("采购项", "包装"),
            (("采购项甲", "密封"),),
        ),
        (
            ("候选人", "文件名"),
            (("候选人X", "10_简历.pdf"), ("候选人Y", "2_简历.pdf")),
            "按文件名排一下",
            ("候选人", "文件名"),
            (("候选人Y", "2_简历.pdf"), ("候选人X", "10_简历.pdf")),
        ),
        (
            ("条款", "版本"),
            (("条款甲", "第12版"), ("条款乙", "第3版")),
            "按版本排",
            ("条款", "版本"),
            (("条款乙", "第3版"), ("条款甲", "第12版")),
        ),
        (
            ("采购项", "批次"),
            (("采购项甲", "批次2"), ("采购项乙", "批次11")),
            "按批次倒序排吧",
            ("采购项", "批次"),
            (("采购项乙", "批次11"), ("采购项甲", "批次2")),
        ),
    ],
)
def test_existing_structured_table_refinement_is_domain_neutral_and_local(
    columns,
    rows,
    question,
    expected_columns,
    expected_rows,
):
    original = StructuredTable(columns=columns, rows=rows)

    refinement = refine_structured_table(original, question)

    assert refinement.valid is True
    assert refinement.table.columns == expected_columns
    assert refinement.table.rows == expected_rows
    assert sorted(cell for row in refinement.table.rows for cell in row) == sorted(
        cell for row in original.rows for cell in row
    )
    if question == "采购项列宽调到16":
        assert refinement.options.column_widths == (("采购项", 16),)


@pytest.mark.parametrize(
    ("columns", "rows", "question", "expected_first"),
    [
        (
            ("候选对象", "年限"),
            (("候选对象10", "三年"), ("候选对象2", "五年")),
            "按候选对象名称排序",
            "候选对象2",
        ),
        (
            ("合同条款", "期限"),
            (("合同条款12", "十日"), ("合同条款3", "三十日")),
            "按合同条款名排序",
            "合同条款3",
        ),
        (
            ("采购项", "数量"),
            (("采购项11", "一件"), ("采购项2", "两件")),
            "按采购项名称排列",
            "采购项2",
        ),
        (
            ("文件", "发布方", "官方性"),
            (
                ("10_合成资料.md", "组织甲", "高"),
                ("2_合成资料.md", "组织乙", "低"),
            ),
            "可以按文件名排列吗？",
            "2_合成资料.md",
        ),
    ],
)
def test_natural_nominal_reference_uniquely_resolves_an_existing_column(
    columns,
    rows,
    question,
    expected_first,
):
    refinement = refine_structured_table(
        StructuredTable(columns=columns, rows=rows),
        question,
    )

    assert refinement.valid is True
    assert refinement.table.rows[0][0] == expected_first


@pytest.mark.parametrize(
    ("columns", "question"),
    [
        (("对象", "对象名称"), "按对象名排序"),
        (("条目", "说明"), "按价格名称排序"),
    ],
    ids=["ambiguous-existing-columns", "new-factual-column"],
)
def test_natural_column_reference_fails_safe_when_not_unique_or_not_existing(
    columns,
    question,
):
    table = StructuredTable(
        columns=columns,
        rows=(tuple(f"合成值{index}" for index in range(len(columns))),),
    )

    refinement = refine_structured_table(table, question)

    assert refinement.valid is False
    assert refinement.table is None


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("可以整齐一点吗？", TableRenderOptions(balanced=True, wrap=True)),
        ("紧凑一点", TableRenderOptions(compact=True)),
        ("不要换行", TableRenderOptions(wrap=False)),
        (
            "列宽调到12",
            TableRenderOptions(column_widths=(("对象", 12), ("说明", 12))),
        ),
    ],
)
def test_layout_only_refinement_updates_bounded_render_options(question, expected):
    table = StructuredTable(
        columns=("对象", "说明"),
        rows=(("合成项甲", "已有说明"),),
    )

    refinement = refine_structured_table(table, question)

    assert refinement.valid is True
    assert refinement.table == table
    assert refinement.options == expected
    assert "合成项甲" in render_structured_table(
        refinement.table,
        refinement.options,
    )


@pytest.mark.parametrize(
    "question",
    [
        "整齐一点并补充薪资",
        "按不存在的字段排序",
        "按不存在的字段排一下",
        "新增一列风险等级",
        "按说明排一下并告诉我出处",
        "把采购项改成采购项乙",
        "列宽调到10000",
    ],
)
def test_local_refinement_rejects_new_facts_unknown_columns_and_unsafe_bounds(question):
    table = StructuredTable(
        columns=("对象", "说明"),
        rows=(("合成项甲", "已有说明"),),
    )

    refinement = refine_structured_table(table, question)

    assert refinement.valid is False
    assert refinement.table is None
    assert refinement.options is None


@pytest.mark.parametrize(
    ("previous_answer", "columns", "row"),
    [
        (
            "候选人甲掌握接口开发。",
            ["候选对象", "能力维度"],
            ["候选人甲", "接口开发"],
        ),
        (
            "条款甲约定交付日期。",
            ["合同项目", "履约信息"],
            ["条款甲", "交付日期"],
        ),
        (
            "采购项甲要求密封包装。",
            ["采购对象", "验收维度"],
            ["采购项甲", "密封包装"],
        ),
    ],
)
def test_column_scaffolding_can_abstract_existing_facts_across_domains(
    previous_answer,
    columns,
    row,
):
    assert all(column not in previous_answer for column in columns)

    validation = validate_table_presentation(
        {"columns": columns, "rows": [row]},
        previous_answer=previous_answer,
    )

    assert validation.valid is True


@pytest.mark.parametrize(
    "question",
    ["给我个表格吧", "整理成表格", "用 Markdown 表格"],
)
def test_table_execution_subtype_matches_table_requests(question):
    assert is_table_presentation_request(question) is True


@pytest.mark.parametrize("question", ["换成列表", "简短一点", "做成三列"])
def test_table_execution_subtype_does_not_capture_adjacent_presentations(question):
    assert is_table_presentation_request(question) is False


def test_provider_config_separates_google_shape_from_strict_internal_schema():
    original = types.GenerateContentConfig(temperature=0.4)

    structured = build_table_generation_config(original)

    assert original.response_schema is None
    assert original.max_output_tokens is None
    assert structured.response_mime_type == "application/json"
    assert structured.response_schema == {
        "type": "object",
        "properties": {
            "columns": {"type": "array", "items": {"type": "string"}},
            "rows": {
                "type": "array",
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
        "required": ["columns", "rows"],
    }
    assert TablePresentationSchema.model_json_schema()["additionalProperties"] is False
    assert structured.max_output_tokens == MAX_TABLE_OUTPUT_TOKENS
    assert structured.temperature == 0.0


def test_current_google_sdk_constructs_production_table_schema_without_unsupported_fields(
    monkeypatch,
):
    client = genai.Client(api_key="synthetic-key", vertexai=False)
    captured = {}
    response_body = json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    '{"columns":["项目"],'
                                    '"rows":[["合成项甲"]]}'
                                )
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ]
        }
    )

    def capture_request(http_request, _http_options=None, stream=False):
        assert stream is False
        captured.update(http_request.data)
        return SimpleNamespace(headers={}, response_stream=[response_body])

    monkeypatch.setattr(client._api_client, "_request", capture_request)
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="仅转换上一回答。",
            config=build_table_generation_config(
                types.GenerateContentConfig(temperature=0.4)
            ),
        )
    finally:
        client.close()

    wire_schema = captured["generationConfig"]["responseSchema"]

    def schema_keys(value):
        if isinstance(value, dict):
            for key, nested in value.items():
                yield key
                yield from schema_keys(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from schema_keys(nested)

    assert "additional_properties" not in set(schema_keys(wire_schema))
    assert "additionalProperties" not in set(schema_keys(wire_schema))
    assert wire_schema["required"] == ["columns", "rows"]
    assert response.parsed == {"columns": ["项目"], "rows": [["合成项甲"]]}


def test_json_fallback_is_still_locally_schema_validated():
    validation = parse_table_presentation(
        '{"columns":["项目"],"rows":[["合成项甲"]]}',
        previous_answer="已有合成项甲。",
    )

    assert validation.valid is True
    assert validation.table.rows == (("合成项甲",),)


@pytest.mark.parametrize(
    ("payload", "previous_answer", "reason"),
    [
        ({"columns": [], "rows": [["甲"]]}, "甲", "columns_empty"),
        ({"columns": ["项目"], "rows": []}, "甲", "rows_empty"),
        ({"columns": ["项目"], "rows": "甲"}, "甲", "rows_not_array"),
        ({"columns": ["项目", "内容"], "rows": [["甲"]]}, "甲", "row_width"),
        ({"columns": ["项目"], "rows": [[1]]}, "1", "cell_type"),
        (
            {"columns": ["项目"], "rows": [["甲"]], "unexpected": "乙"},
            "甲乙",
            "unexpected_fields",
        ),
        (
            {
                "columns": [f"列{index}" for index in range(MAX_TABLE_COLUMNS + 1)],
                "rows": [["甲"] * (MAX_TABLE_COLUMNS + 1)],
            },
            "甲",
            "columns_limit",
        ),
        (
            {"columns": ["项目"], "rows": [["甲"]] * (MAX_TABLE_ROWS + 1)},
            "甲",
            "rows_limit",
        ),
        (
            {"columns": ["项目"], "rows": [["甲" * (MAX_TABLE_CELL_CHARS + 1)]]},
            "甲" * (MAX_TABLE_CELL_CHARS + 1),
            "cell_size_limit",
        ),
        (
            {"columns": ["列" * 81], "rows": [["甲"]]},
            "甲",
            "column_size_limit",
        ),
        ({"columns": ["项目"], "rows": [[""]]}, "甲", "empty_content"),
    ],
)
def test_malformed_or_empty_structured_table_fails_closed(
    payload,
    previous_answer,
    reason,
):
    validation = validate_table_presentation(
        payload,
        previous_answer=previous_answer,
    )

    assert validation.valid is False
    assert validation.table is None
    assert validation.reason == reason


def test_total_character_limit_is_enforced_after_per_cell_validation():
    cell = "甲" * MAX_TABLE_CELL_CHARS
    validation = validate_table_presentation(
        {"columns": ["项目"], "rows": [[cell]] * 33},
        previous_answer=cell,
    )

    assert validation.valid is False
    assert validation.reason == "total_size_limit"


@pytest.mark.parametrize(
    ("previous_answer", "columns", "row"),
    [
        (
            "候选人甲掌握接口开发。",
            ["候选对象", "能力维度"],
            ["候选人甲", "五年经验"],
        ),
        (
            "条款甲约定交付日期。",
            ["合同项目", "履约信息"],
            ["条款甲", "违约金"],
        ),
        (
            "采购项甲要求密封包装。",
            ["采购对象", "验收维度"],
            ["采购项甲", "次日到货"],
        ),
    ],
)
def test_factual_cells_remain_previous_answer_bound_across_domains(
    previous_answer,
    columns,
    row,
):
    validation = validate_table_presentation(
        {"columns": columns, "rows": [row]},
        previous_answer=previous_answer,
    )

    assert validation.valid is False
    assert validation.reason == "cell_not_in_previous_answer"
