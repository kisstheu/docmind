from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from app.dialog_state_machine import ConversationState
from app.domain_host import StaticDomainHost
from bootstrap.domain_composition import create_domain_host
from docmind_domain_sdk import DomainRequest, validate_result_boundary
from docmind_recruitment_plugin import PLUGIN_ID, RecruitmentJDPlugin
from public_tests.domain.test_domain_dispatch_port_empty_host import (
    _run_turn,
    _stale_interaction_state,
)


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

NON_RECRUITMENT_TEXT = (
    "项目需求说明：本项目在示例地点实施，交付要求包括接口文档、测试记录和回滚方案，"
    "相关经验只用于安排完全合成的项目任务。"
)


def test_real_runner_presents_handled_markdown_once_and_resets_core_state(
    monkeypatch,
    tmp_path,
) -> None:
    network_attempts = []

    def fail_on_network(*args, **kwargs):
        network_attempts.append((args, kwargs))
        raise AssertionError("recruitment slice must not access the network")

    monkeypatch.setattr("socket.socket.connect", fail_on_network)
    plugin = RecruitmentJDPlugin()
    execute_spy = AsyncMock(wraps=plugin.execute)
    monkeypatch.setattr(plugin, "execute", execute_spy)
    host = create_domain_host(plugin=plugin, expected_plugin_id=PLUGIN_ID)

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=[STANDARD_JD],
        initial_state=_stale_interaction_state(),
        initial_focus="scope-a.md",
        use_real_dialog_events=True,
    )

    assert isinstance(host, StaticDomainHost)
    execute_spy.assert_awaited_once()
    request = execute_spy.await_args.args[0]
    assert isinstance(request, DomainRequest)
    assert request.query == STANDARD_JD
    assert request.source_scope == ()
    result = asyncio.run(RecruitmentJDPlugin().execute(request))
    validate_result_boundary(request, result, expected_plugin_id=PLUGIN_ID)
    assert result.status == "handled"
    assert result.focus_update.mode == "preserve"
    assert result.focus_update.items == ()
    assert result.focus_update.selected is None
    assert result.evidence == ()
    assert result.warnings == ()
    assert result.error is None
    assert captured["printed"] == [result.answer_markdown]
    assert captured["memory_calls"] == [(request.query, result.answer_markdown)]
    assert captured["memory_snapshots"] == [[
        f"用户问：{request.query}",
        f"AI答：{result.answer_markdown}",
    ]]
    assert captured["search"] == []
    assert captured["materials"] == []
    assert captured["prompts"] == []
    assert captured["state_updates"] == []
    assert fake_models.calls == []
    assert network_attempts == []
    assert state == ConversationState()


def test_real_runner_falls_back_unchanged_after_recruitment_plugin_abstains(
    monkeypatch,
    tmp_path,
) -> None:
    plugin = RecruitmentJDPlugin()
    execute_spy = AsyncMock(wraps=plugin.execute)
    monkeypatch.setattr(plugin, "execute", execute_spy)
    host = create_domain_host(plugin=plugin, expected_plugin_id=PLUGIN_ID)
    initial_state = _stale_interaction_state()

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=[NON_RECRUITMENT_TEXT],
        initial_state=initial_state,
        initial_focus="scope-a.md",
    )

    execute_spy.assert_awaited_once()
    request = execute_spy.await_args.args[0]
    assert request.query == NON_RECRUITMENT_TEXT
    assert request.source_scope == ()
    direct_result = asyncio.run(RecruitmentJDPlugin().execute(request))
    assert direct_result.status == "abstain"
    assert direct_result.answer_markdown == ""
    assert direct_result.evidence == ()
    assert direct_result.warnings == ()
    assert direct_result.error is None
    assert len(captured["search"]) == 1
    assert len(captured["materials"]) == 1
    assert captured["search"][0]["question"] == request.query
    assert captured["materials"][0]["question"] == request.query
    assert captured["materials"][0]["current_focus_file"] == "scope-a.md"
    assert captured["printed"] == ["受控本地结果"]
    assert captured["memory_calls"] == [(request.query, "受控本地结果")]
    assert captured["state_updates"] == ["受控本地结果"]
    assert captured["prompts"] == []
    assert fake_models.calls == []
    assert state.last_route == "normal_retrieval"
    assert state.last_result_set_items == ["scope-a.md", "scope-b.md"]
    assert state.last_selected_candidate == "候选项X"

    direct_host = create_domain_host(
        plugin=RecruitmentJDPlugin(),
        expected_plugin_id=PLUGIN_ID,
    )
    assert direct_host.dispatch(request) is None
