from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai.decision_result import parse_decision_result, render_decision_result
from app import chat_loop as chat_runtime
import app.chat_loop_parts.runner as chat_runner
from app.dialog.state_machine import ConversationState, detect_dialog_event
from app.dialog.task_semantics import classify_answer_mode, is_recommendation_request
from app.domain_host.host import EmptyDomainHost
from public_tests.state.test_repo_meta_generic_file_list_scope_contract import (
    _EmbeddingStub,
    _RecordingLogger,
    _indexed_repo_state,
)


_DOMAINS = [
    ("采购说明.md", "收货前", "抽检"),
    ("合同说明.md", "签署前", "复核"),
    ("课程说明.md", "开课前", "预习"),
]


@pytest.mark.parametrize("source,when,operation", _DOMAINS)
@pytest.mark.parametrize("polar", ["是否", "有没有", "有无"])
def test_polar_recommendation_is_evidence_not_candidate_selection(source, when, operation, polar):
    question = f"根据{source}，{when}{polar}推荐常规{operation}？仅在哪两类情况下需要{operation}？"
    assert not is_recommendation_request(question)
    assert classify_answer_mode(question) == "evidence"
    event = detect_dialog_event(question, ConversationState(), _RecordingLogger())
    assert event.name != "decision_request"


@pytest.mark.parametrize(
    "question",
    [
        "请推荐一个方案", "是否推荐一个方案？", "有没有推荐给我的方案？",
        "根据资料，是否推荐常规复核？另外请推荐一个方案。",
        "根据资料，是否推荐常规复核？帮我选一份。",
        "根据资料给我推荐一份", "推荐方案A还是方案B？",
        "哪个方案更适合？", "比较方案A和方案B",
    ],
)
def test_explicit_selection_and_comparison_keep_decision_contract(question):
    assert classify_answer_mode(question) == "decision"
    assert detect_dialog_event(question, ConversationState(), _RecordingLogger()).name == "decision_request"


@pytest.mark.parametrize("source,when,operation", _DOMAINS)
@pytest.mark.parametrize("delivery", ["local", "generated"])
def test_evidence_survives_real_chat_delivery_without_selection_state(
    monkeypatch, tmp_path, capsys, source, when, operation, delivery,
):
    question = f"根据{source}，{when}是否推荐常规{operation}？仅在哪两类情况下需要{operation}？"
    if delivery == "local":
        question = f"是否推荐常规{operation}？"
    facts = f"不推荐常规{operation}。仅以下两类情况需要{operation}：①出现异常；②约定明确要求。"
    answer = f"{facts}\n来源文件：{source}"
    repo = _indexed_repo_state([source], [facts])
    inputs = iter([question, "exit"])
    calls = []

    def generate_content(*, model, contents, config=None):
        assert delivery == "generated", "Complete local evidence must not invoke remote generation"
        calls.append(contents)
        assert facts in contents
        assert "【比较与决策任务】" not in contents
        return SimpleNamespace(text=answer)

    monkeypatch.setattr(chat_runtime, "_read_user_question", lambda **_kwargs: next(inputs))
    monkeypatch.setattr(chat_runtime, "_flush_pending_tty_input_unix", lambda: False)
    monkeypatch.setattr(chat_runtime, "conversation_state", ConversationState())
    monkeypatch.setattr(
        "app.retrieval_flow.query.rewrite_search_query",
        lambda *_args, **_kwargs: operation if delivery == "local" else question,
    )
    def forbidden_decision_parser(*_args, **_kwargs):
        pytest.fail("Factual answers must not be consumed as candidate decisions")

    monkeypatch.setattr(chat_runner, "parse_decision_result", forbidden_decision_parser)
    chat_runtime.run_chat_loop(
        repo, _EmbeddingStub(), SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
        "offline-model", "http://127.0.0.1:9", "offline-model", _RecordingLogger(),
        notes_dir=tmp_path / "notes", change_log_file=tmp_path / "changes.db",
        domain_dispatch_port=EmptyDomainHost(),
    )
    output = capsys.readouterr().out
    assert facts in output
    assert source in output
    assert "目前没有足够匹配的候选" not in output
    assert len(calls) == (1 if delivery == "generated" else 0)
    assert chat_runtime.conversation_state.last_selected_candidate is None
    assert not chat_runtime.conversation_state.last_selected_source_files


@pytest.mark.parametrize("source,when,operation", _DOMAINS)
def test_valid_single_candidate_keeps_selection(source, when, operation):
    result = parse_decision_result(
        f"推荐对象：方案A\n推荐理由：支持{when}{operation}\n来源文件：{source}",
        user_question="推荐一个方案",
    )
    assert result is not None and result.has_selection
    assert result.selected_candidate == "方案A"
    assert result.source_files == (source,)
    assert "方案A" in render_decision_result(result)


@pytest.mark.parametrize(
    "answer",
    [
        "推荐结论：暂无足够匹配的候选",
        "推荐对象：方案A",  # Unsupported: no source.
        "推荐对象：根据证据所以推荐以下对象\n来源文件：合成资料.md",
        "推荐对象：方案A；方案B\n来源文件：合成资料.md",
        "推荐结论：证据不足\n推荐对象：方案A\n来源文件：合成资料.md",
        "来源文件：合成资料.md",  # A source alone is not a selection.
    ],
)
def test_actual_decisions_still_fail_closed_without_valid_candidate(answer):
    result = parse_decision_result(answer, user_question="推荐一个方案")
    assert result is not None and not result.has_selection
    assert result.selected_candidate is None
    assert result.source_files == ()
    assert render_decision_result(result) == "目前没有足够匹配的候选。"
