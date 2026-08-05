from __future__ import annotations

import ast
import asyncio
import inspect
import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

import docmind_recruitment_plugin.plugin as plugin_module

from app.domain_host import EmptyDomainHost, StaticDomainHost
from bootstrap.domain_composition import create_domain_host
from docmind_domain_sdk import (
    DomainPlugin,
    DomainRequest,
    PluginDescribeRequest,
    PluginStartRequest,
    PluginStopRequest,
    SourceSyncRequest,
    validate_lifecycle_boundary,
    validate_manifest_boundary,
    validate_probe_boundary,
    validate_result_boundary,
    validate_sync_boundary,
)
from docmind_recruitment_plugin import (
    DISPLAY_NAME,
    PLUGIN_ID,
    PLUGIN_VERSION,
    RecruitmentJDPlugin,
)


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PACKAGE_ROOT / "src" / "docmind_recruitment_plugin"

STANDARD_JD = """请整理这份 JD 的明确约束
职位名称：Python AI 应用开发工程师
岗位职责：负责某公司A内部知识库、RAG 检索流程、后端接口和自动化测试体系开发，维护稳定交付流程并编写清晰技术文档。
任职要求：
1. 熟悉 Python，并能编写可维护代码；
2. 了解 FastAPI；
3. 具有 RAG 项目经验；
4. 掌握 SQL。
工作地点：示例城市甲
工作制：双休
经验要求：2 年以上
薪资：15K–20K"""

PARTIAL_JD = """岗位名称：后端开发工程师
岗位职责：负责完全合成的订单接口、服务稳定性、日志排查和自动化交付，持续维护技术文档并按明确接口约定协作。
技术要求：熟悉 Python；掌握 SQL；了解 FastAPI。
办公地点：示例城市甲
工作经验：3 年以上"""

ALTERNATE_JD = """提取下面岗位中明确写出的条件
招聘职位：数据接口开发工程师
工作内容：建设完全合成的数据接口与质量检查流程，维护可重复执行的交付记录和清晰的服务文档。
技能要求：掌握 Java；熟悉 Spring Boot；了解 SQL。
薪酬范围：薪资面议
工作制度：周一至周五
学历要求：本科及以上"""


def _request(
    query: str = "Synthetic ordinary text.",
    *,
    options=None,
) -> DomainRequest:
    return DomainRequest(
        request_id="b2-request",
        query=query,
        source_scope=(),
        **({} if options is None else {"options": options}),
    )


def _execute(query: str, *, options=None):
    return asyncio.run(
        RecruitmentJDPlugin().execute(_request(query, options=options))
    )


def _recruitment_options(rules: dict[str, object]):
    return {
        PLUGIN_ID: {
            "schema_version": "1.0",
            "explicit_rules": rules,
        }
    }


def _jd_at_non_whitespace_length(target: int) -> str:
    template = (
        "职位名称：合成开发工程师\n岗位职责：{padding}\n技术要求：熟悉 Python"
        "\n工作地点：示例城市甲\n经验要求：2年以上"
    )
    without_padding = template.format(padding="")
    base_length = sum(not character.isspace() for character in without_padding)
    assert target >= base_length
    return template.format(padding="甲" * (target - base_length))


def _assert_abstain(result) -> None:
    assert result.status == "abstain"
    assert result.answer_markdown == ""
    assert result.focus_update.mode == "preserve"
    assert result.focus_update.items == ()
    assert result.focus_update.selected is None
    assert result.evidence == ()
    assert result.warnings == ()
    assert result.error is None


def _assert_handled_profile(result) -> None:
    assert result.status == "handled"
    assert result.answer_markdown.startswith("## JD 明确约束\n\n")
    assert result.answer_markdown.endswith(
        "> 仅整理原文明确内容；未列出的字段表示原文没有明确说明。"
    )
    assert result.focus_update.mode == "preserve"
    assert result.focus_update.items == ()
    assert result.focus_update.selected is None
    assert result.evidence == ()
    assert result.warnings == ()
    assert result.error is None


def test_public_package_import_and_protocol_shape() -> None:
    plugin = RecruitmentJDPlugin()
    assert isinstance(plugin, DomainPlugin)
    for method_name in ("describe", "start", "sync_sources", "probe", "execute", "stop"):
        assert inspect.iscoroutinefunction(getattr(plugin, method_name))


def test_manifest_is_stable_and_uses_no_permissions() -> None:
    plugin = RecruitmentJDPlugin()
    request = PluginDescribeRequest(request_id="describe-1")
    result = asyncio.run(plugin.describe(request))

    validate_manifest_boundary(request, result, expected_plugin_id=PLUGIN_ID)
    assert PLUGIN_ID == "org.docmind.recruitment.jd-constraints"
    assert PLUGIN_VERSION == "0.1.0"
    assert result.display_name == DISPLAY_NAME
    assert result.schema_version == "1.1"
    assert result.transport_modes == ("in_process",)
    assert result.permissions == ()


def test_lifecycle_and_source_sync_return_valid_dtos() -> None:
    plugin = RecruitmentJDPlugin()
    start_request = PluginStartRequest(
        request_id="start-1",
        host_instance_id="host-1",
    )
    start_result = asyncio.run(plugin.start(start_request))
    validate_lifecycle_boundary(
        start_request,
        start_result,
        expected_plugin_id=PLUGIN_ID,
    )
    assert start_result.status == "ok"

    sync_request = SourceSyncRequest(request_id="sync-1")
    sync_result = asyncio.run(plugin.sync_sources(sync_request))
    validate_sync_boundary(sync_request, sync_result, expected_plugin_id=PLUGIN_ID)
    assert sync_result.status == "ok"
    assert sync_result.accepted_source_ids == ()
    assert sync_result.rejected_source_ids == ()

    stop_request = PluginStopRequest(
        request_id="stop-1",
        host_instance_id="host-1",
    )
    stop_result = asyncio.run(plugin.stop(stop_request))
    validate_lifecycle_boundary(
        stop_request,
        stop_result,
        expected_plugin_id=PLUGIN_ID,
    )
    assert stop_result.status == "ok"


def test_probe_claims_only_supported_structure_without_extracting_fields() -> None:
    plugin = RecruitmentJDPlugin()
    supported_request = _request(STANDARD_JD)
    supported = asyncio.run(plugin.probe(supported_request))
    validate_probe_boundary(
        supported_request,
        supported,
        expected_plugin_id=PLUGIN_ID,
    )
    assert supported.disposition == "claim"
    assert supported.score == 1.0
    assert supported.evidence_source_refs == ()

    unsupported_request = _request()
    unsupported = asyncio.run(plugin.probe(unsupported_request))
    validate_probe_boundary(
        unsupported_request,
        unsupported,
        expected_plugin_id=PLUGIN_ID,
    )
    assert unsupported.disposition == "abstain"
    assert unsupported.score == 0.0
    assert unsupported.evidence_source_refs == ()


def test_probe_can_claim_structure_that_execute_still_rejects() -> None:
    query = (
        "职位名称：合成工程师\n"
        "岗位职责：" + "甲" * 80 + "\n"
        "任职要求：具备良好沟通能力；工作认真负责。"
    )
    request = _request(query)
    plugin = RecruitmentJDPlugin()

    probe = asyncio.run(plugin.probe(request))
    result = asyncio.run(plugin.execute(request))

    assert probe.disposition == "claim"
    _assert_abstain(result)


@pytest.mark.parametrize("query", [STANDARD_JD, PARTIAL_JD, ALTERNATE_JD])
def test_supported_structured_jd_is_handled(query: str) -> None:
    request = _request(query)
    result = asyncio.run(RecruitmentJDPlugin().execute(request))

    validate_result_boundary(request, result, expected_plugin_id=PLUGIN_ID)
    _assert_handled_profile(result)


@pytest.mark.parametrize(
    "title_label",
    ("岗位名称", "职位名称", "招聘岗位", "招聘职位"),
)
def test_all_approved_title_labels_are_supported(title_label: str) -> None:
    query = STANDARD_JD.replace("职位名称：", f"{title_label}：", 1)

    _assert_handled_profile(_execute(query))


def test_unapproved_title_label_abstains() -> None:
    query = STANDARD_JD.replace("职位名称：", "应聘岗位：", 1)

    _assert_abstain(_execute(query))


@pytest.mark.parametrize(
    "requirement_label",
    (
        "任职要求",
        "岗位要求",
        "职位要求",
        "技能要求",
        "技术要求",
        "任职资格",
        "岗位资格",
    ),
)
def test_all_approved_requirement_labels_are_supported(
    requirement_label: str,
) -> None:
    query = STANDARD_JD.replace("任职要求：", f"{requirement_label}：", 1)

    _assert_handled_profile(_execute(query))


def test_standard_jd_renders_only_stable_ordered_original_fields() -> None:
    result = _execute(STANDARD_JD)

    assert result.answer_markdown == """## JD 明确约束

- 岗位名称：Python AI 应用开发工程师
- 薪资：15K–20K
- 工作地点：示例城市甲
- 工作制：双休
- 经验：2 年以上
- 技术要求：熟悉 Python；并能编写可维护代码；了解 FastAPI；具有 RAG 项目经验；掌握 SQL

> 仅整理原文明确内容；未列出的字段表示原文没有明确说明。"""


def test_all_explicit_fields_keep_qualifiers_and_render_in_fixed_order() -> None:
    query = STANDARD_JD.replace(
        "工作地点：示例城市甲",
        "工作地点：示例城市甲\n加班情况：偶尔加班\n是否外包：否\n是否驻场：每月驻场 2 天\n学历要求：本科及以上",
    )
    result = _execute(query)

    lines = [line for line in result.answer_markdown.splitlines() if line.startswith("- ")]
    assert lines == [
        "- 岗位名称：Python AI 应用开发工程师",
        "- 薪资：15K–20K",
        "- 工作地点：示例城市甲",
        "- 工作制：双休",
        "- 加班或大小周：偶尔加班",
        "- 外包：否",
        "- 驻场：每月驻场 2 天",
        "- 学历：本科及以上",
        "- 经验：2 年以上",
        "- 技术要求：熟悉 Python；并能编写可维护代码；了解 FastAPI；具有 RAG 项目经验；掌握 SQL",
    ]


def test_partial_jd_at_four_field_threshold_is_handled_without_missing_fields() -> None:
    result = _execute(PARTIAL_JD)

    _assert_handled_profile(result)
    assert [line.split("：", 1)[0] for line in result.answer_markdown.splitlines() if line.startswith("- ")] == [
        "- 岗位名称",
        "- 工作地点",
        "- 经验",
        "- 技术要求",
    ]
    assert "薪资：" not in result.answer_markdown
    assert "外包：" not in result.answer_markdown
    assert "驻场：" not in result.answer_markdown
    assert "未提及" not in result.answer_markdown


def test_repeated_equivalent_title_is_one_job_and_uses_first_original_value() -> None:
    query = STANDARD_JD.replace(
        "职位名称：Python AI 应用开发工程师",
        "职位名称：Python AI 应用开发工程师\n岗位名称：python-ai/应用开发工程师",
    )
    result = _execute(query)

    _assert_handled_profile(result)
    assert "- 岗位名称：Python AI 应用开发工程师" in result.answer_markdown


def test_two_complete_blocks_with_same_title_still_abstain() -> None:
    second_block = """

岗位名称：Python AI 应用开发工程师
岗位职责：负责另一个完全合成的服务交付块。
技术要求：熟悉 Go；了解 Redis。
工作地点：示例城市甲
工作制：双休"""

    _assert_abstain(_execute(STANDARD_JD + second_block))


def test_repeated_requirement_sections_in_one_job_remain_one_jd() -> None:
    query = STANDARD_JD.replace(
        "4. 掌握 SQL。",
        "4. 掌握 SQL。\n技能要求：使用 Docker；了解 Go。",
    )

    result = _execute(query)

    _assert_handled_profile(result)
    assert "掌握 SQL；使用 Docker；了解 Go" in result.answer_markdown


@pytest.mark.parametrize(
    "query",
    [
        STANDARD_JD.replace(
            "岗位职责：",
            "岗位名称：前端开发工程师\n岗位职责：",
        ),
        STANDARD_JD
        + "\n\n职位名称：后端开发工程师\n岗位职责：维护另一套合成服务。"
        + "\n任职要求：熟悉 Go；了解 Redis。\n工作地点：示例城市甲\n工作制：双休",
    ],
)
def test_distinct_titles_or_two_complete_jd_blocks_abstain(query: str) -> None:
    _assert_abstain(_execute(query))


@pytest.mark.parametrize(
    "query",
    [
        "这是一段普通说明，不是招聘信息。",
        "职位名称：后端开发工程师",
        STANDARD_JD.replace("任职要求：", "能力说明："),
        "职位名称：合成工程师\n任职要求：熟悉 Python\n工作地点：示例城市甲",
        STANDARD_JD + "\n这个岗位值不值得去？",
        STANDARD_JD + "\n请结合候选人X的简历和偏好判断是否适合我。",
        STANDARD_JD + "\n请与另一个岗位比较并排序。",
        STANDARD_JD + "\n请访问 https://example.invalid/job 补充信息。",
        STANDARD_JD + "\n请列出这个岗位的风险和红旗。",
        STANDARD_JD + "\n请调查这家公司背景。",
        STANDARD_JD + "\n请匹配我的简历。",
        STANDARD_JD + "\n请读取已检索文件补充信息。",
        STANDARD_JD + "\n请给出投递建议。",
        STANDARD_JD + "\n请给出面试建议。",
        STANDARD_JD + "\n请给出薪资谈判建议。",
        (
            "岗位名称：合成开发工程师\n岗位职责：负责完全合成的内部服务和文档维护，"
            "按既定流程交付并记录每次变更。\n任职要求：熟悉 Python；了解 FastAPI。"
            "\n工作地点：示例城市甲"
        ),
        (
            "岗位名称：合成运营工程师\n岗位职责：负责完全合成的流程维护、记录整理和服务协调，"
            "确保交付材料清晰完整。\n任职要求：具备良好沟通能力；工作认真负责。"
            "\n工作地点：示例城市甲\n工作制：双休\n学历要求：本科及以上\n经验要求：2 年以上"
        ),
    ],
)
def test_unsupported_or_incomplete_input_returns_pure_abstain(query: str) -> None:
    _assert_abstain(_execute(query))


def test_technical_requirements_preserve_qualifiers_order_and_deduplicate() -> None:
    query = PARTIAL_JD.replace(
        "熟悉 Python；掌握 SQL；了解 FastAPI",
        "必须熟悉 Python；了解 FastAPI；优先掌握 SQL；了解 FastAPI",
    )
    result = _execute(query)

    assert "- 技术要求：必须熟悉 Python；了解 FastAPI；优先掌握 SQL" in result.answer_markdown


def test_technical_rule_handles_chinese_and_varied_latin_technologies() -> None:
    query = STANDARD_JD.replace(
        "1. 熟悉 Python，并能编写可维护代码；\n"
        "2. 了解 FastAPI；\n"
        "3. 具有 RAG 项目经验；\n"
        "4. 掌握 SQL。",
        "1. 熟悉 Java；\n"
        "2. 了解 Go；\n"
        "3. 使用 Docker；\n"
        "4. 掌握数据库索引优化；\n"
        "5. 具备良好沟通能力。",
    )

    result = _execute(query)

    technical_line = next(
        line for line in result.answer_markdown.splitlines() if line.startswith("- 技术要求：")
    )
    assert technical_line == (
        "- 技术要求：熟悉 Java；了解 Go；使用 Docker；掌握数据库索引优化"
    )


def test_explicit_technical_section_keeps_chinese_original_phrases() -> None:
    query = PARTIAL_JD.replace(
        "熟悉 Python；掌握 SQL；了解 FastAPI",
        "掌握数据库索引优化；了解服务部署流程；具备良好沟通能力",
    )

    result = _execute(query)

    assert "- 技术要求：掌握数据库索引优化；了解服务部署流程" in result.answer_markdown
    assert "沟通能力" not in result.answer_markdown


@pytest.mark.parametrize(
    "technical_text",
    (
        "good communication",
        "fluent English",
        "teamwork and leadership",
        "具有丰富项目经验",
        "具备良好沟通能力；工作认真负责",
    ),
)
def test_soft_skills_and_ordinary_english_are_not_technical_requirements(
    technical_text: str,
) -> None:
    query = PARTIAL_JD.replace(
        "熟悉 Python；掌握 SQL；了解 FastAPI",
        technical_text,
    )

    _assert_abstain(_execute(query))


def test_explicit_skill_section_does_not_turn_communication_into_technical_requirement() -> None:
    query = PARTIAL_JD.replace(
        "熟悉 Python；掌握 SQL；了解 FastAPI",
        "熟悉 Python；具备良好沟通能力；掌握 SQL",
    )
    result = _execute(query)

    technical_line = next(
        line for line in result.answer_markdown.splitlines() if line.startswith("- 技术要求：")
    )
    assert technical_line == "- 技术要求：熟悉 Python；掌握 SQL"


def test_comma_separated_soft_clause_is_excluded_from_technical_requirement() -> None:
    query = PARTIAL_JD.replace(
        "熟悉 Python；掌握 SQL；了解 FastAPI",
        "熟悉 Python，具备良好沟通能力；掌握 SQL",
    )

    result = _execute(query)

    technical_line = next(
        line for line in result.answer_markdown.splitlines() if line.startswith("- 技术要求：")
    )
    assert technical_line == "- 技术要求：熟悉 Python；掌握 SQL"


def test_duties_are_not_inferred_as_technical_requirements() -> None:
    query = STANDARD_JD.replace(
        "负责某公司A内部知识库、RAG 检索流程、后端接口和自动化测试体系开发",
        "负责 Go 服务和 Redis 集群的日常维护",
    ).replace("具有 RAG 项目经验；\n", "")
    result = _execute(query)

    _assert_handled_profile(result)
    technical_line = next(
        line for line in result.answer_markdown.splitlines() if line.startswith("- 技术要求：")
    )
    assert "Go" not in technical_line
    assert "Redis" not in technical_line
    assert "Python" in technical_line


def test_title_technology_is_not_inferred_when_requirement_section_is_soft_only() -> None:
    query = STANDARD_JD.replace(
        "1. 熟悉 Python，并能编写可维护代码；\n"
        "2. 了解 FastAPI；\n"
        "3. 具有 RAG 项目经验；\n"
        "4. 掌握 SQL。",
        "1. 具备良好沟通能力；\n2. 工作认真负责。",
    )

    _assert_abstain(_execute(query))


def test_conflicting_non_title_field_is_rendered_without_counting_as_valid() -> None:
    query = STANDARD_JD.replace("工作制：双休", "工作制：双休\n工作制度：单休")

    result = _execute(query)

    _assert_handled_profile(result)
    assert "- 工作制：双休；单休（原文存在冲突）" in result.answer_markdown


def test_execute_is_deterministic_and_profiles_are_strict() -> None:
    first = _execute(STANDARD_JD)
    second = _execute(STANDARD_JD)

    assert first == second
    _assert_handled_profile(first)
    _assert_abstain(_execute("Synthetic ordinary text."))


def test_non_whitespace_length_boundary_is_exact_and_not_the_only_gate() -> None:
    below = _jd_at_non_whitespace_length(79)
    boundary = _jd_at_non_whitespace_length(80)

    assert sum(not character.isspace() for character in below) == 79
    assert sum(not character.isspace() for character in boundary) == 80
    _assert_abstain(_execute(below))
    _assert_handled_profile(_execute(boundary))
    _assert_abstain(
        _execute("甲" * 80 + "\n职位名称：合成开发工程师\n工作地点：示例城市甲")
    )


def test_approved_instruction_is_excluded_from_exact_length_boundary() -> None:
    instruction = "请整理这份 JD 的明确约束"

    for query in (
        instruction + "\n" + _jd_at_non_whitespace_length(79),
        _jd_at_non_whitespace_length(79) + "\n" + instruction,
    ):
        _assert_abstain(_execute(query))
    for query in (
        instruction + "\n" + _jd_at_non_whitespace_length(80),
        _jd_at_non_whitespace_length(80) + "\n" + instruction,
    ):
        _assert_handled_profile(_execute(query))


def test_more_than_one_approved_instruction_abstains() -> None:
    body = STANDARD_JD.split("\n", 1)[1]
    query = (
        "请整理这份 JD 的明确约束\n"
        + body
        + "\n提取下面岗位中明确写出的条件"
    )

    _assert_abstain(_execute(query))


@pytest.mark.parametrize(
    "query",
    [
        "招聘宣传：某公司A持续招聘技术人才，欢迎关注。职位名称：开发工程师。",
        "候选人X简历：工作经验三年，熟悉 Python，期望地点为示例城市甲。",
        "面试复盘：面试中讨论了项目经验和工作地点，后续继续准备。",
        "求职偏好说明：希望双休、地点方便，并关注成长空间。",
        "合同条款：履约要求为按期交付，工作地点以书面通知为准。",
        "采购规格：技术要求包含接口文档，经验参数仅用于供应商说明。",
        "项目需求说明：开发地点为示例机房，要求提交测试记录。",
        "技术方案：实施要求包含回滚方案和接口验证，不涉及招聘。",
        "培训课程介绍：学习要求为具备基础经验，地点为示例教室。",
    ],
)
def test_adjacent_and_cross_domain_samples_never_handle(query: str) -> None:
    _assert_abstain(_execute(query))


def test_execute_does_not_access_network(monkeypatch: pytest.MonkeyPatch) -> None:
    network_attempts = []

    def fail_on_network(*args, **kwargs):
        network_attempts.append((args, kwargs))
        raise AssertionError("plugin skeleton must not access the network")

    monkeypatch.setattr("socket.socket.connect", fail_on_network)
    result = asyncio.run(RecruitmentJDPlugin().execute(_request(STANDARD_JD)))

    assert result.status == "handled"
    assert network_attempts == []


def test_plugin_source_uses_only_sdk_top_level_and_no_core_imports() -> None:
    imports = []
    for source in sorted(SOURCE_ROOT.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")

    assert "docmind_domain_sdk" in imports
    assert all(not name.startswith("docmind_domain_sdk.") for name in imports)
    forbidden_roots = {"app", "bootstrap", "ai", "retrieval", "infra", "ask_notes"}
    assert forbidden_roots.isdisjoint(name.split(".", 1)[0] for name in imports)
    external_roots = {"requests", "httpx", "urllib", "socket", "google", "openai"}
    assert external_roots.isdisjoint(name.split(".", 1)[0] for name in imports)

    extraction_tree = ast.parse(
        (SOURCE_ROOT / "extraction.py").read_text(encoding="utf-8")
    )
    extraction_literals = (
        node.value
        for node in ast.walk(extraction_tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    )
    target_stack_terms = (
        "Python",
        "FastAPI",
        "RAG",
        "Java",
        "Go",
        "Docker",
        "Redis",
        "SQL",
    )
    target_stack_pattern = re.compile(
        r"(?<![A-Za-z0-9])(?:"
        + "|".join(re.escape(term) for term in target_stack_terms)
        + r")(?![A-Za-z0-9])"
    )
    assert all(target_stack_pattern.search(value) is None for value in extraction_literals)


def test_factory_injects_plugin_once_and_maps_abstain_to_none(monkeypatch) -> None:
    plugin = RecruitmentJDPlugin()
    execute_spy = AsyncMock(wraps=plugin.execute)
    monkeypatch.setattr(plugin, "execute", execute_spy)
    host = create_domain_host(plugin=plugin, expected_plugin_id=PLUGIN_ID)

    assert isinstance(host, StaticDomainHost)
    assert host.dispatch(_request()) is None
    execute_spy.assert_awaited_once()


def test_existing_empty_host_contract_remains_available() -> None:
    host = create_domain_host()

    assert isinstance(host, EmptyDomainHost)
    assert host.dispatch(_request()) is None


def test_no_options_empty_options_other_namespace_and_empty_rules_are_exact_noops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("no-op paths must not enter comparison rendering")

    monkeypatch.setattr(plugin_module, "compare_job_search_rules", forbidden)
    monkeypatch.setattr(plugin_module, "render_job_rule_comparison", forbidden)
    baseline = _execute(STANDARD_JD)
    variants = (
        _execute(STANDARD_JD, options={}),
        _execute(
            STANDARD_JD,
            options={"org.example.contract": {"synthetic": "value"}},
        ),
        _execute(STANDARD_JD, options=_recruitment_options({})),
    )

    assert all(result == baseline for result in variants)
    assert all(result.answer_markdown == baseline.answer_markdown for result in variants)
    assert baseline.answer_markdown.startswith("## JD 明确约束\n\n")


@pytest.mark.parametrize(
    ("rules", "rendered_label"),
    (
        ({"minimum_monthly_salary_k": 14}, "- 薪资："),
        ({"require_double_weekends": True}, "- 工作制："),
        ({"allow_outsourcing": False}, "- 外包："),
        ({"allow_onsite": False}, "- 驻场："),
        ({"allowed_locations": ["示例城市甲"]}, "- 工作地点："),
        ({"candidate_education_level": "associate"}, "- 学历："),
        ({"candidate_relevant_years": 3}, "- 经验年限："),
    ),
)
def test_each_structured_rule_field_individually_reaches_existing_comparison(
    rules: dict[str, object],
    rendered_label: str,
) -> None:
    result = _execute(STANDARD_JD, options=_recruitment_options(rules))

    assert result.status == "handled"
    assert result.answer_markdown.startswith("## 单 JD 显式规则比较\n\n")
    assert result.answer_markdown.count(rendered_label) == 1
    assert result.evidence == ()
    assert result.warnings == ()
    assert result.error is None


def test_combined_rules_reuse_match_conflict_and_unknown_rendering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compare_calls = []
    render_calls = []
    original_compare = plugin_module.compare_job_search_rules
    original_render = plugin_module.render_job_rule_comparison

    def capture_compare(*args, **kwargs):
        compare_calls.append((args, kwargs))
        return original_compare(*args, **kwargs)

    def capture_render(*args, **kwargs):
        render_calls.append((args, kwargs))
        return original_render(*args, **kwargs)

    monkeypatch.setattr(
        plugin_module,
        "compare_job_search_rules",
        capture_compare,
    )
    monkeypatch.setattr(
        plugin_module,
        "render_job_rule_comparison",
        capture_render,
    )
    query = STANDARD_JD.replace(
        "工作地点：示例城市甲",
        "工作地点：示例城市甲\n是否外包：第三方签约\n是否驻场：需要现场沟通",
    )
    result = _execute(
        query,
        options=_recruitment_options(
            {
                "minimum_monthly_salary_k": 14,
                "allow_outsourcing": False,
                "allow_onsite": False,
            }
        ),
    )

    markdown = result.answer_markdown
    assert markdown.startswith("## 单 JD 显式规则比较\n\n")
    assert markdown.index("### 明确符合") < markdown.index("- 薪资：")
    assert markdown.index("### 明确冲突") < markdown.index("- 外包：")
    assert markdown.index("### 信息缺失或需要确认") < markdown.index("- 驻场：")
    assert markdown.index("- 薪资：") < markdown.index("### 明确冲突")
    assert markdown.index("- 外包：") < markdown.index("### 信息缺失或需要确认")
    assert len(compare_calls) == 1
    assert len(render_calls) == 1


@pytest.mark.parametrize(
    ("query", "rules", "label"),
    (
        (STANDARD_JD, {"allow_outsourcing": False}, "- 外包："),
        (
            STANDARD_JD.replace(
                "工作地点：示例城市甲",
                "工作地点：示例城市甲\n是否驻场：需要现场沟通",
            ),
            {"allow_onsite": False},
            "- 驻场：",
        ),
        (
            STANDARD_JD.replace(
                "工作制：双休",
                "工作制：双休\n工作制度：单休",
            ),
            {"require_double_weekends": True},
            "- 工作制：",
        ),
    ),
)
def test_missing_ambiguous_and_source_conflict_are_unknown_not_conflict(
    query: str,
    rules: dict[str, object],
    label: str,
) -> None:
    markdown = _execute(
        query,
        options=_recruitment_options(rules),
    ).answer_markdown

    unknown_start = markdown.index("### 信息缺失或需要确认")
    technical_start = markdown.index("### 技术要求原文")
    assert label in markdown[unknown_start:technical_start]
    conflict_section = markdown[
        markdown.index("### 明确冲突"):unknown_start
    ]
    assert conflict_section == "### 明确冲突\n- 无\n\n"


def test_invalid_own_namespace_returns_fixed_redacted_handled_result() -> None:
    request = _request(
        STANDARD_JD,
        options=_recruitment_options(
            {
                "minimum_monthly_salary_k": 987654.25,
                "synthetic_unknown_rule": "示例城市隐私哨兵",
            }
        ),
    )
    query_before = request.query
    result = asyncio.run(RecruitmentJDPlugin().execute(request))

    assert result.status == "handled"
    assert result.answer_markdown == """## 求职规则输入无效

本次未执行显式规则比较。请检查结构化求职规则后重试。"""
    assert result.focus_update.mode == "preserve"
    assert result.focus_update.items == ()
    assert result.focus_update.selected is None
    assert result.evidence == ()
    assert result.warnings == ()
    assert result.error is None
    assert request.query == query_before
    for sentinel in (
        PLUGIN_ID,
        "minimum_monthly_salary_k",
        "synthetic_unknown_rule",
        "987654.25",
        "示例城市隐私哨兵",
        "RecruitmentOptionsError",
    ):
        assert sentinel not in result.answer_markdown


def test_invalid_payload_never_calls_comparison_or_comparison_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison_calls = []
    rendering_calls = []

    def forbidden_comparison(*args, **kwargs):
        comparison_calls.append((args, kwargs))
        raise AssertionError("invalid payload must not compare")

    def forbidden_rendering(*args, **kwargs):
        rendering_calls.append((args, kwargs))
        raise AssertionError("invalid payload must not render comparison")

    monkeypatch.setattr(
        plugin_module,
        "compare_job_search_rules",
        forbidden_comparison,
    )
    monkeypatch.setattr(
        plugin_module,
        "render_job_rule_comparison",
        forbidden_rendering,
    )

    result = _execute(
        STANDARD_JD,
        options=_recruitment_options(
            {
                "require_double_weekends": True,
                "synthetic_unknown_rule": False,
            }
        ),
    )

    assert result.answer_markdown.startswith("## 求职规则输入无效")
    assert comparison_calls == []
    assert rendering_calls == []
    assert "明确符合" not in result.answer_markdown


@pytest.mark.parametrize(
    "query",
    (
        "招聘宣传：某公司A持续招聘技术人才，欢迎关注。职位名称：开发工程师。",
        "合同条款：履约要求为按期交付，工作地点以书面通知为准。",
        "采购规格：技术要求包含接口文档，经验参数仅用于供应商说明。",
        "项目需求说明：开发地点为示例机房，要求提交测试记录。",
    ),
)
def test_invalid_options_do_not_change_adjacent_or_cross_domain_abstain(
    query: str,
) -> None:
    request = _request(
        query,
        options={PLUGIN_ID: {"synthetic_unknown_namespace_key": True}},
    )
    query_before = request.query

    result = asyncio.run(RecruitmentJDPlugin().execute(request))

    _assert_abstain(result)
    assert request.query == query_before


def test_invalid_options_do_not_change_multiple_jd_abstain() -> None:
    second = STANDARD_JD.replace(
        "Python AI 应用开发工程师",
        "合成数据开发工程师",
        1,
    )
    query = STANDARD_JD + "\n\n" + second

    result = _execute(
        query,
        options={PLUGIN_ID: {"synthetic_unknown_namespace_key": True}},
    )

    _assert_abstain(result)


def test_query_is_exactly_preserved_across_success_noop_rejection_and_abstain() -> None:
    cases = (
        (STANDARD_JD, _recruitment_options({"minimum_monthly_salary_k": 14})),
        (STANDARD_JD, _recruitment_options({})),
        (STANDARD_JD, {PLUGIN_ID: None}),
        ("这是一段完全合成的普通项目说明。", {PLUGIN_ID: None}),
    )

    for index, (query, options) in enumerate(cases):
        request = DomainRequest(
            request_id=f"query-preservation-{index}",
            query=query,
            source_scope=(),
            options=options,
        )
        before = request.query
        asyncio.run(RecruitmentJDPlugin().execute(request))
        assert request.query == before


def test_empty_location_array_has_no_effective_rule_and_keeps_legacy_output() -> None:
    baseline = _execute(STANDARD_JD)
    result = _execute(
        STANDARD_JD,
        options=_recruitment_options({"allowed_locations": []}),
    )

    assert result == baseline
