"""A non-empty direct answer is final only when it covers the whole request."""

from types import SimpleNamespace

import pytest

from public_tests.retrieval.test_direct_factual_evidence_contract import answer


CASES = [
    (
        "课程甲的开始时间、持续时长和及格标准分别是多少？",
        "课程甲 开始时间 持续时长 及格标准",
        "课程甲\n开始时间为上午9点。",
        "课程甲\n开始时间为上午9点，持续时长为45分钟，及格标准为72分。",
    ),
    (
        "商品甲的交付时间、包装件数和合格阈值分别是多少？",
        "商品甲 交付时间 包装件数 合格阈值",
        "商品甲\n交付时间为确认后8天。",
        "商品甲\n交付时间为确认后8天，包装件数为24件，合格阈值为96%。",
    ),
    (
        "行星甲的公转周期、卫星数量和表面温度分别是多少？",
        "行星甲 公转周期 卫星数量 表面温度",
        "行星甲\n公转周期为400天。",
        "行星甲\n公转周期为400天，卫星数量为2颗，表面温度为20℃。",
    ),
]


@pytest.mark.parametrize("question,query,partial,complete", CASES)
def test_partial_multi_fact_local_evidence_hands_off(question, query, partial, complete):
    assert answer(
        question, query, ["资料甲.md"], [partial], force_local_evidence=True,
    ) is None


@pytest.mark.parametrize("question,query,partial,complete", CASES)
def test_complete_multi_fact_local_evidence_remains_final(question, query, partial, complete):
    result = answer(
        question, query, ["资料甲.md"], [complete], force_local_evidence=True,
    )
    assert result is not None
    assert complete.splitlines()[-1] in result
    assert "来源：资料甲.md" in result


@pytest.mark.parametrize("question,query,partial,complete", CASES)
def test_no_local_answer_preserves_normal_fallback(question, query, partial, complete):
    assert answer(
        question, query, ["资料甲.md"], ["这里只记录无关背景。"], force_local_evidence=True,
    ) is None


def test_partial_local_evidence_reaches_existing_generation_path_once(
    monkeypatch, tmp_path, capsys,
):
    from app import chat_loop as runtime
    from app.chat_loop_parts import runner
    from app.dialog.state_machine import ConversationState
    from app.domain_host.host import EmptyDomainHost
    from public_tests.state.test_repo_meta_generic_file_list_scope_contract import (
        _EmbeddingStub,
        _RecordingLogger,
        _indexed_repo_state,
    )

    question, query, partial, _ = CASES[1]
    generated = "根据资料，交付时间为确认后8天；其他请求项尚需补充证据。来源：资料甲.md"
    repo = _indexed_repo_state(["资料甲.md"], [partial])
    inputs = iter([question, "exit"])
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        assert partial.splitlines()[-1] in kwargs["contents"]
        return SimpleNamespace(text=generated)

    monkeypatch.setenv("DOCMIND_EVIDENCE_SCOPE_BINDING", "0")
    monkeypatch.setattr(runtime, "_read_user_question", lambda **_kwargs: next(inputs))
    monkeypatch.setattr(runtime, "_flush_pending_tty_input_unix", lambda: False)
    monkeypatch.setattr(runtime, "conversation_state", ConversationState())
    monkeypatch.setattr("app.retrieval_flow.query.rewrite_search_query", lambda *_args, **_kwargs: query)
    logger = _RecordingLogger()
    monkeypatch.setattr(
        runner,
        "resolve_route",
        lambda question, event, *_args, **_kwargs: {
            "route": "normal_retrieval",
            "smalltalk_reply": "",
            "route_question_input": question,
        },
    )
    runtime.run_chat_loop(
        repo,
        _EmbeddingStub(),
        SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
        "offline-model",
        "http://127.0.0.1:9",
        "offline-model",
        logger,
        notes_dir=tmp_path / "notes",
        change_log_file=tmp_path / "changes.db",
        domain_dispatch_port=EmptyDomainHost(),
    )

    output = capsys.readouterr().out
    assert len(calls) == 1
    assert generated in output
    assert any("[远程模型生成]" in message for message in logger.messages)
    assert any("[本地充分性守门]" in message for message in logger.messages)
    assert "先给你可直接核对的证据" not in output


def test_complete_local_evidence_does_not_call_remote(monkeypatch, tmp_path, capsys):
    from app import chat_loop as runtime
    from app.chat_loop_parts import runner
    from app.dialog.state_machine import ConversationState
    from app.domain_host.host import EmptyDomainHost
    from public_tests.state.test_repo_meta_generic_file_list_scope_contract import (
        _EmbeddingStub,
        _RecordingLogger,
        _indexed_repo_state,
    )

    question = "按2042年版采购说明，到货时间和抽检比例分别是多少？"
    query = "2042年版采购说明 到货时间 抽检比例"
    complete = "执行前先核对已确认条件，到货时间为10至20天，抽检比例为10%至20%。"
    repo = _indexed_repo_state(["资料甲.md"], [complete])
    inputs = iter([question, "exit"])

    def forbidden_remote(**_kwargs):
        pytest.fail("Sufficient local evidence must retain final authority")

    monkeypatch.setattr(runtime, "_read_user_question", lambda **_kwargs: next(inputs))
    monkeypatch.setattr(runtime, "_flush_pending_tty_input_unix", lambda: False)
    monkeypatch.setattr(runtime, "conversation_state", ConversationState())
    monkeypatch.setattr("app.retrieval_flow.query.rewrite_search_query", lambda *_args, **_kwargs: query)
    monkeypatch.setattr(
        runner,
        "resolve_route",
        lambda current_question, event, *_args, **_kwargs: {
            "route": "normal_retrieval",
            "smalltalk_reply": "",
            "route_question_input": current_question,
        },
    )
    runtime.run_chat_loop(
        repo,
        _EmbeddingStub(),
        SimpleNamespace(models=SimpleNamespace(generate_content=forbidden_remote)),
        "offline-model",
        "http://127.0.0.1:9",
        "offline-model",
        _RecordingLogger(),
        notes_dir=tmp_path / "notes",
        change_log_file=tmp_path / "changes.db",
        domain_dispatch_port=EmptyDomainHost(),
    )

    output = capsys.readouterr().out
    assert complete.splitlines()[-1] in output
    assert "来源：资料甲.md" in output
    assert "[远程模型生成]" not in output
