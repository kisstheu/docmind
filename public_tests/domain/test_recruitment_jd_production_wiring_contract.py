from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from docmind_domain_sdk import DomainRequest

import ask_notes
from app.domain_host import StaticDomainHost
from app.domain_dispatch_port import dispatch_domain_request
from docmind_recruitment_plugin import PLUGIN_ID, RecruitmentJDPlugin


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REQUIREMENTS_FILE = REPOSITORY_ROOT / "requirements.txt"

SYNTHETIC_JD = """请整理这份 JD 中明确写出的岗位约束
职位名称：Python AI 应用开发工程师
岗位职责：负责示例公司甲的内部知识库、接口和测试流程，维护可重复执行的交付记录与技术文档。
任职要求：
1. 熟悉 Python；
2. 了解 FastAPI；
3. 具有 RAG 项目经验；
4. 掌握 SQL。
工作地点：示例城市甲
工作制：双休
经验要求：2 年以上
薪资：15K–20K"""


def _request(query: str) -> DomainRequest:
    return DomainRequest(
        request_id="production-wiring-request",
        query=query,
        source_scope=(),
    )


def test_requirements_keeps_utf16_crlf_and_unique_local_package_entries() -> None:
    data = REQUIREMENTS_FILE.read_bytes()

    assert data.startswith(b"\xff\xfe")
    assert data.count(b"\n\x00") == data.count(b"\r\x00\n\x00")
    text = data.decode("utf-16")
    assert "\ufffd" not in text
    lines = text.splitlines()
    assert lines.count("./packages/docmind-domain-sdk") == 1
    assert lines.count("./packages/docmind-recruitment-plugin") == 1
    local_package_entries = [
        line for line in lines if line.startswith("./packages/docmind-")
    ]
    assert len(local_package_entries) == len(set(local_package_entries))


def test_production_composition_constructs_plugin_and_calls_factory_once(
    monkeypatch,
) -> None:
    plugin = object()
    host = object()
    plugin_calls = []
    factory_calls = []

    def construct_plugin():
        plugin_calls.append(True)
        return plugin

    def capture_factory(**kwargs):
        factory_calls.append(kwargs)
        return host

    monkeypatch.setattr(ask_notes, "RecruitmentJDPlugin", construct_plugin)
    monkeypatch.setattr(ask_notes, "create_domain_host", capture_factory)

    assert ask_notes.create_production_domain_host() is host
    assert plugin_calls == [True]
    assert factory_calls == [
        {
            "plugin": plugin,
            "expected_plugin_id": PLUGIN_ID,
        }
    ]


def test_default_production_host_handles_supported_jd_without_network(
    monkeypatch,
) -> None:
    network_attempts = []

    def fail_on_network(*args, **kwargs):
        network_attempts.append((args, kwargs))
        raise AssertionError("production recruitment host must not access the network")

    monkeypatch.setattr("socket.socket.connect", fail_on_network)
    host = ask_notes.create_production_domain_host()

    assert isinstance(host, StaticDomainHost)
    result = host.dispatch(_request(SYNTHETIC_JD))
    assert result is not None
    assert result.status == "handled"
    assert "- 岗位名称：Python AI 应用开发工程师" in result.answer_markdown
    assert "- 工作地点：示例城市甲" in result.answer_markdown
    assert "- 技术要求：" in result.answer_markdown
    assert result.focus_update.mode == "preserve"
    assert result.focus_update.items == ()
    assert result.focus_update.selected is None
    assert result.evidence == ()
    assert result.warnings == ()
    assert result.error is None
    assert network_attempts == []


def test_default_production_host_maps_plugin_abstain_to_none() -> None:
    host = ask_notes.create_production_domain_host()

    assert isinstance(host, StaticDomainHost)
    assert host.dispatch(_request("这是一段完全合成的普通项目说明。")) is None


def test_main_uses_production_composition_helper() -> None:
    tree = ast.parse(Path(ask_notes.__file__).read_text(encoding="utf-8"))
    main_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    called_names = [
        node.func.id
        for node in ast.walk(main_function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]

    assert called_names.count("create_production_domain_host") == 1
    assert "create_domain_host" not in called_names
    assert called_names.count("run_chat_loop") == 1


def _recruitment_options(rules: dict[str, object]):
    return {
        PLUGIN_ID: {
            "schema_version": "1.0",
            "explicit_rules": rules,
        }
    }


def _load_cli_options(monkeypatch, tmp_path, options):
    path = tmp_path / "synthetic-domain-options.json"
    path.write_text(json.dumps(options, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["docmind-test", "--domain-options-file", str(path)],
    )
    return ask_notes._parse_args().domain_options


def test_cli_loader_to_production_host_reaches_existing_comparison_markdown(
    monkeypatch,
    tmp_path,
) -> None:
    options = _recruitment_options(
        {
            "minimum_monthly_salary_k": 21,
            "require_double_weekends": True,
            "allow_outsourcing": False,
        }
    )
    options["org.example.synthetic"] = {"marker": "coexisting-namespace"}
    snapshot = _load_cli_options(monkeypatch, tmp_path, options)
    host = ask_notes.create_production_domain_host()

    result = dispatch_domain_request(
        host,
        SYNTHETIC_JD,
        options=snapshot,
    )

    assert isinstance(host, StaticDomainHost)
    assert result is not None
    assert result.status == "handled"
    assert result.answer_markdown.startswith("## 单 JD 显式规则比较\n\n")
    assert "- 薪资：" in result.answer_markdown
    assert "- 工作制：" in result.answer_markdown
    assert "- 外包：" in result.answer_markdown
    assert "### 明确符合" in result.answer_markdown
    assert "### 明确冲突" in result.answer_markdown
    assert "### 信息缺失或需要确认" in result.answer_markdown
    match_section, remainder = result.answer_markdown.split("### 明确冲突", 1)
    conflict_section, unknown_section = remainder.split("### 信息缺失或需要确认", 1)
    assert "- 工作制：JD 明确为双休，满足显式规则。" in match_section
    assert "- 薪资：JD 月薪上限 20K 低于显式规则下限 21K。" in conflict_section
    assert "- 外包：JD 未明确是否外包。" in unknown_section
    assert result.evidence == ()
    assert result.warnings == ()
    assert result.error is None


def test_production_dispatch_transparently_preserves_query_and_options(
    monkeypatch,
    tmp_path,
) -> None:
    observed = []
    original_execute = RecruitmentJDPlugin.execute

    async def capture_execute(self, request):
        observed.append((request.query, request.options))
        return await original_execute(self, request)

    monkeypatch.setattr(RecruitmentJDPlugin, "execute", capture_execute)
    options = _load_cli_options(
        monkeypatch,
        tmp_path,
        _recruitment_options(
            {
                "allowed_locations": ["示例城市甲"],
                "candidate_education_level": "associate",
                "candidate_relevant_years": 3,
            }
        ),
    )
    host = ask_notes.create_production_domain_host()

    result = dispatch_domain_request(host, SYNTHETIC_JD, options=options)

    assert result is not None
    assert result.answer_markdown.startswith("## 单 JD 显式规则比较\n\n")
    assert observed == [(SYNTHETIC_JD, options)]


def test_production_dispatch_no_options_and_other_namespace_are_exact_noops(
    monkeypatch,
    tmp_path,
) -> None:
    fixed_id = type("FixedRequestId", (), {"hex": "production-noop-request"})()
    monkeypatch.setattr(
        "app.domain_dispatch_port.uuid4",
        lambda: fixed_id,
    )
    host = ask_notes.create_production_domain_host()
    monkeypatch.setattr(sys, "argv", ["docmind-test"])
    omitted = ask_notes._parse_args().domain_options
    empty_options = _load_cli_options(monkeypatch, tmp_path, {})
    other_options = _load_cli_options(
        monkeypatch,
        tmp_path,
        {"org.example.contract": {"synthetic": "value"}},
    )

    baseline = dispatch_domain_request(host, SYNTHETIC_JD) if omitted is None else None
    empty = dispatch_domain_request(host, SYNTHETIC_JD, options=empty_options)
    other = dispatch_domain_request(
        host,
        SYNTHETIC_JD,
        options=other_options,
    )

    assert baseline is not None
    assert empty == baseline
    assert other == baseline
    assert baseline.answer_markdown.startswith("## JD 明确约束\n\n")


def test_cli_accepts_generic_but_business_invalid_payload_for_plugin_rejection(
    monkeypatch,
    tmp_path,
    capsys,
    caplog,
) -> None:
    options = _recruitment_options(
        {
            "minimum_monthly_salary_k": 987654.25,
            "synthetic_unknown_rule": "示例隐私哨兵",
        }
    )
    snapshot = _load_cli_options(monkeypatch, tmp_path, options)
    assert snapshot == options
    host = ask_notes.create_production_domain_host()

    result = dispatch_domain_request(
        host,
        SYNTHETIC_JD,
        options=snapshot,
    )

    assert result is not None
    assert result.status == "handled"
    assert result.answer_markdown == """## 求职规则输入无效

本次未执行显式规则比较。请检查结构化求职规则后重试。"""
    assert result.evidence == ()
    assert result.warnings == ()
    assert result.error is None
    output = capsys.readouterr()
    assert output.out == output.err == ""
    for sentinel in (
        PLUGIN_ID,
        "minimum_monthly_salary_k",
        "synthetic_unknown_rule",
        "987654.25",
        "示例隐私哨兵",
    ):
        assert sentinel not in result.answer_markdown
        assert sentinel not in caplog.text


def test_recruitment_options_do_not_claim_adjacent_cross_domain_or_multiple_jd(
    monkeypatch,
    tmp_path,
) -> None:
    snapshot = _load_cli_options(
        monkeypatch,
        tmp_path,
        _recruitment_options({"require_double_weekends": True}),
    )
    second_jd = SYNTHETIC_JD.replace(
        "Python AI 应用开发工程师",
        "合成数据开发工程师",
        1,
    )
    queries = (
        "这是一段完全合成的普通说明。",
        "合同条款：履约要求为按期交付，工作地点以书面通知为准。",
        "采购规格：技术要求包含接口文档，经验参数仅用于供应商说明。",
        "项目需求说明：开发地点为示例机房，要求提交测试记录。",
        SYNTHETIC_JD + "\n\n" + second_jd,
    )
    host = ask_notes.create_production_domain_host()

    results = [
        dispatch_domain_request(host, query, options=snapshot)
        for query in queries
    ]

    assert results == [None] * len(queries)


def test_options_payload_and_sentinel_never_reach_output_logs_or_abstain_result(
    monkeypatch,
    tmp_path,
    capsys,
    caplog,
) -> None:
    sentinel = "HIGH-DISCLOSURE-SYNTHETIC-SENTINEL"
    snapshot = _load_cli_options(
        monkeypatch,
        tmp_path,
        {"org.example.synthetic": {"marker": sentinel}},
    )
    host = ask_notes.create_production_domain_host()

    result = dispatch_domain_request(
        host,
        "合同条款：本合成文本只描述交付记录。",
        options=snapshot,
    )

    output = capsys.readouterr()
    assert result is None
    assert output.out == output.err == ""
    assert sentinel not in caplog.text
    assert "org.example.synthetic" not in caplog.text
