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


SINGLE_FACT_CASES = [
    (
        "根据合同甲说明，合同甲是否允许自动续期？",
        "合同甲",
        "合同甲一般归档记录完整。",
        "合同甲不允许自动续期。",
    ),
    (
        "根据商品甲说明，商品甲是否支持次日退换？",
        "商品甲",
        "商品甲一般包装清单已经复核。",
        "商品甲支持次日退换。",
    ),
    (
        "根据行星甲资料，行星甲是否拥有两颗卫星？",
        "行星甲",
        "行星甲一般观测档案已经归档。",
        "行星甲拥有两颗卫星。",
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


@pytest.mark.parametrize("question,query,irrelevant,complete", SINGLE_FACT_CASES)
def test_irrelevant_high_score_single_fact_candidate_hands_off(
    question, query, irrelevant, complete,
):
    result = answer(
        question,
        query,
        ["资料甲.md", "资料甲.md"],
        [irrelevant, complete],
        force_local_evidence=True,
    )
    assert result is None


@pytest.mark.parametrize("question,query,irrelevant,complete", SINGLE_FACT_CASES)
def test_complete_relevant_single_fact_candidate_remains_local(
    question, query, irrelevant, complete,
):
    result = answer(
        question,
        query,
        ["资料甲.md"],
        [complete],
        force_local_evidence=True,
    )
    assert result is not None
    assert complete in result
    assert "来源：资料甲.md" in result


def test_explicit_source_scope_rejects_cross_source_contamination():
    result = answer(
        "根据资料甲.md，课程甲是否允许补考？",
        "课程甲",
        ["资料甲.md", "资料乙.md"],
        ["课程甲允许补考。", "课程甲的作业允许延期提交。"],
        force_local_evidence=True,
    )
    assert result is None


def test_explicit_source_scope_keeps_complete_bound_answer():
    result = answer(
        "根据资料甲.md，课程甲是否允许补考？",
        "课程甲",
        ["资料甲.md", "资料乙.md"],
        ["课程甲允许补考。", "课程甲的作业允许延期提交。"],
        indices=[0],
        force_local_evidence=True,
    )
    assert result is not None
    assert "课程甲允许补考。" in result
    assert "来源：资料甲.md" in result
    assert "资料乙.md" not in result


def test_weak_rewrite_cannot_finalize_two_clause_polar_request():
    question = "根据2042年版采购说明，商品甲是否推荐常规抽检？仅在哪两类情况下需要抽检？"
    complete = (
        "商品甲不推荐常规抽检，仅以下情况需要抽检：①外包装存在异常记录；"
        "②双方约定明确要求抽检。"
    )
    result = answer(
        question,
        "商品甲",
        ["采购说明_2042年版.md", "采购说明_2042年版.md"],
        ["商品甲一般事项已归档。", complete],
        force_local_evidence=True,
    )
    assert result is None


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


def test_irrelevant_single_fact_local_candidate_reaches_generation_once(
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

    question, query, irrelevant, complete = SINGLE_FACT_CASES[0]
    generated = f"{complete}\n来源：资料甲.md"
    repo = _indexed_repo_state(["资料甲.md"], [f"{irrelevant}\n{complete}"])
    inputs = iter([question, "exit"])
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        assert complete in kwargs["contents"]
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
        lambda current_question, event, *_args, **_kwargs: {
            "route": "normal_retrieval",
            "smalltalk_reply": "",
            "route_question_input": current_question,
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
    assert irrelevant not in output
