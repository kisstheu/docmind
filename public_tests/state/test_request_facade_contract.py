from dataclasses import asdict
from types import SimpleNamespace

import pytest

from app import chat_loop
from app.chat_loop_parts import runner
from app.dialog_state_machine import ConversationState
from app.request_facade import SingleRequestPort
from public_tests.state.test_core_comparison_result_presentation_contract import _fixture
from public_tests.state.test_repo_meta_generic_file_list_scope_contract import _run_turns


@pytest.mark.parametrize('kind', ['设备', '合同方案', '岗位'])
@pytest.mark.parametrize('failure', [False, True])
def test_request_port_preserves_core_output_and_guards(monkeypatch, tmp_path, kind, failure):
    paths, facts, _, response = _fixture(kind, '离线运行', '条件待确认', '必须联网')
    question = f'比较这几个{kind}并推荐一个，说明下一步。'
    port = SingleRequestPort(question)
    original_run = chat_loop.run_chat_loop
    monkeypatch.setattr(chat_loop, 'run_chat_loop', lambda *a, **kw: original_run(*a, **kw, request_port=port))

    def forbidden(*args, **kwargs):
        raise AssertionError('programmatic read-only requests must not execute file actions')

    monkeypatch.setattr(runner, 'handle_file_action_turn', forbidden)
    client = SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: SimpleNamespace(text='' if failure else response)))
    _run_turns(monkeypatch, tmp_path, questions=[], repo_paths=paths, repo_chunks=facts,
               state=ConversationState(), client=client)
    assert port.result['ok'] is not failure
    if not failure:
        assert port.result['answer'] == chat_loop.conversation_state.last_answer_text
        assert port.result['decision']['source_files'] == tuple(paths)
        table = chat_loop.conversation_state.current_presentation_table
        assert port.result['table_presentation'] == (asdict(table) if table else None)
    else:
        assert 'answer' not in port.result


def test_port_does_not_export_old_answer_when_validation_fails():
    port = SingleRequestPort('合成请求')
    port.finish_turn(state=ConversationState(last_answer_text='上一轮成果'), decision_result=None, valid=False)
    assert not port.result['ok'] and 'answer' not in port.result
    assert port.read_question() == '合成请求'
    assert port.read_question() == 'exit'
