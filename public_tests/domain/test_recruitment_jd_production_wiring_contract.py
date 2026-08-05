from __future__ import annotations

import ast
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


def test_production_dispatch_reaches_existing_comparison_markdown() -> None:
    host = ask_notes.create_production_domain_host()

    result = dispatch_domain_request(
        host,
        SYNTHETIC_JD,
        options=_recruitment_options(
            {
                "minimum_monthly_salary_k": 14,
                "require_double_weekends": True,
                "allow_outsourcing": False,
            }
        ),
    )

    assert isinstance(host, StaticDomainHost)
    assert result is not None
    assert result.status == "handled"
    assert result.answer_markdown.startswith("## 单 JD 显式规则比较\n\n")
    assert "- 薪资：" in result.answer_markdown
    assert "- 工作制：" in result.answer_markdown
    assert "- 外包：" in result.answer_markdown
    assert result.evidence == ()
    assert result.warnings == ()
    assert result.error is None


def test_production_dispatch_transparently_preserves_query_and_options(
    monkeypatch,
) -> None:
    observed = []
    original_execute = RecruitmentJDPlugin.execute

    async def capture_execute(self, request):
        observed.append((request.query, request.options))
        return await original_execute(self, request)

    monkeypatch.setattr(RecruitmentJDPlugin, "execute", capture_execute)
    options = _recruitment_options(
        {
            "allowed_locations": ["示例城市甲"],
            "candidate_education_level": "associate",
            "candidate_relevant_years": 3,
        }
    )
    host = ask_notes.create_production_domain_host()

    result = dispatch_domain_request(host, SYNTHETIC_JD, options=options)

    assert result is not None
    assert result.answer_markdown.startswith("## 单 JD 显式规则比较\n\n")
    assert observed == [(SYNTHETIC_JD, options)]


def test_production_dispatch_no_options_and_other_namespace_are_exact_noops(
    monkeypatch,
) -> None:
    fixed_id = type("FixedRequestId", (), {"hex": "production-noop-request"})()
    monkeypatch.setattr(
        "app.domain_dispatch_port.uuid4",
        lambda: fixed_id,
    )
    host = ask_notes.create_production_domain_host()

    baseline = dispatch_domain_request(host, SYNTHETIC_JD)
    empty = dispatch_domain_request(host, SYNTHETIC_JD, options={})
    other = dispatch_domain_request(
        host,
        SYNTHETIC_JD,
        options={"org.example.contract": {"synthetic": "value"}},
    )

    assert baseline is not None
    assert empty == baseline
    assert other == baseline
    assert baseline.answer_markdown.startswith("## JD 明确约束\n\n")


def test_production_dispatch_invalid_payload_returns_fixed_handled_rejection() -> None:
    host = ask_notes.create_production_domain_host()

    result = dispatch_domain_request(
        host,
        SYNTHETIC_JD,
        options=_recruitment_options(
            {
                "minimum_monthly_salary_k": 987654.25,
                "synthetic_unknown_rule": "示例城市隐私哨兵",
            }
        ),
    )

    assert result is not None
    assert result.status == "handled"
    assert result.answer_markdown == """## 求职规则输入无效

本次未执行显式规则比较。请检查结构化求职规则后重试。"""
    assert result.evidence == ()
    assert result.warnings == ()
    assert result.error is None
    for sentinel in (
        PLUGIN_ID,
        "minimum_monthly_salary_k",
        "synthetic_unknown_rule",
        "987654.25",
        "示例城市隐私哨兵",
    ):
        assert sentinel not in result.answer_markdown
