from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

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
PLUGIN_SOURCE = PACKAGE_ROOT / "src" / "docmind_recruitment_plugin" / "plugin.py"


def _request(query: str = "Synthetic ordinary text.") -> DomainRequest:
    return DomainRequest(
        request_id="b2-request",
        query=query,
        source_scope=(),
    )


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
    assert result.schema_version == "1.0"
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


def test_probe_always_abstains_without_claiming_capability() -> None:
    plugin = RecruitmentJDPlugin()
    request = _request()
    result = asyncio.run(plugin.probe(request))

    validate_probe_boundary(request, result, expected_plugin_id=PLUGIN_ID)
    assert result.disposition == "abstain"
    assert result.score == 0.0
    assert result.evidence_source_refs == ()


@pytest.mark.parametrize(
    "query",
    [
        "Synthetic ordinary text.",
        (
            "职位名称：合成软件工程师。岗位职责：维护完全合成的内部工具。"
            "任职要求：熟悉示例语言；能编写自动化测试。工作地点：示例城市。"
        ),
    ],
)
def test_execute_always_returns_minimal_abstain(query: str) -> None:
    plugin = RecruitmentJDPlugin()
    request = _request(query)
    result = asyncio.run(plugin.execute(request))

    validate_result_boundary(request, result, expected_plugin_id=PLUGIN_ID)
    assert result.status == "abstain"
    assert result.answer_markdown == ""
    assert result.focus_update.mode == "preserve"
    assert result.focus_update.items == ()
    assert result.focus_update.selected is None
    assert result.evidence == ()
    assert result.warnings == ()
    assert result.error is None


def test_execute_does_not_access_network(monkeypatch: pytest.MonkeyPatch) -> None:
    network_attempts = []

    def fail_on_network(*args, **kwargs):
        network_attempts.append((args, kwargs))
        raise AssertionError("plugin skeleton must not access the network")

    monkeypatch.setattr("socket.socket.connect", fail_on_network)
    result = asyncio.run(RecruitmentJDPlugin().execute(_request()))

    assert result.status == "abstain"
    assert network_attempts == []


def test_plugin_source_uses_only_sdk_top_level_and_no_core_imports() -> None:
    tree = ast.parse(PLUGIN_SOURCE.read_text(encoding="utf-8"), filename=str(PLUGIN_SOURCE))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    assert set(imports) == {"__future__", "docmind_domain_sdk"}
    assert all(not name.startswith("docmind_domain_sdk.") for name in imports)
    forbidden_roots = {"app", "bootstrap", "ai", "retrieval", "infra", "ask_notes"}
    assert forbidden_roots.isdisjoint(name.split(".", 1)[0] for name in imports)


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
