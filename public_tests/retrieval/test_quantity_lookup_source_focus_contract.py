"""Quantity lookup must focus on source facts instead of a cited edition."""
from types import SimpleNamespace

import pytest

from public_tests.retrieval.test_direct_factual_evidence_contract import answer


DOMAINS = [
    ('采购', '到货时间', '抽检比例', '10 至 20天', '10%–20%'),
    ('合同', '付款金额', '履行期限', '300～500元', '3～5天'),
    ('课程', '班级规模', '练习次数', '10–20人', '3～5次'),
]


def sample(domain, first, second, value_a, value_b):
    question = f'按2042年版{domain}说明，{first}和{second}分别是多少？'
    # The existing rewriter separates these facts; the original question's
    # unsegmented source reference must not become an exclusive focus gate.
    query = f'2042年版{domain}说明 {first} {second}'
    facts = f'执行前先核对已确认条件，{first}为{value_a}，{second}为{value_b}。'
    return question, query, facts


@pytest.mark.parametrize('domain,first,second,value_a,value_b', DOMAINS)
@pytest.mark.parametrize('order', [[0, 1, 2], [2, 1, 0]])
def test_complete_numeric_paragraph_beats_source_only_heading(domain, first, second, value_a, value_b, order):
    q, query, facts = sample(domain, first, second, value_a, value_b)
    result = answer(q, query, ['资料甲.md', '资料乙.md', '资料甲.md'],
                    ['（2042年版）', '（2042年版）', facts], indices=order)
    assert facts in result
    assert '来源：资料甲.md' in result
    assert '资料乙.md' not in result
    assert '（2042年版）' not in result


@pytest.mark.parametrize('domain,first,second,value_a,value_b', DOMAINS)
def test_later_values_in_long_paragraph_are_kept_verbatim(domain, first, second, value_a, value_b):
    q, query, facts = sample(domain, first, second, value_a, value_b)
    paragraph = '执行前需要核对条件、确认记录并保留依据，' * 9 + facts
    assert len(paragraph) > 160
    result = answer(q, query, ['资料甲.md'], [f'（2042年版）\n{paragraph}'])
    assert paragraph in result
    assert value_a in result and value_b in result


@pytest.mark.parametrize('domain,first,second,value_a,value_b', DOMAINS)
def test_single_value_does_not_pull_in_unrelated_paragraphs(domain, first, second, value_a, value_b):
    fact = f'{first}为{value_a}。'
    irrelevant = '附录记录完整讨论过程。' * 30
    result = answer(f'按2042年版{domain}说明，{first}是多少？', f'2042 {first}',
                    ['资料甲.md'], [f'（2042年版）\n{irrelevant}\n{fact}'],
                    force_local_evidence=True)
    assert fact in result
    assert irrelevant not in result
    assert len(result) < 160


@pytest.mark.parametrize('domain,first,second,value_a,value_b', DOMAINS)
@pytest.mark.parametrize('kind', ['heading', 'topic_only', 'wrong_subject', 'outside_scope'])
def test_source_or_topic_without_numeric_body_is_insufficient(domain, first, second, value_a, value_b, kind):
    q, query, facts = sample(domain, first, second, value_a, value_b)
    distractor = {
        'heading': f'（2042年版）\n{first}和{second}。',
        'topic_only': f'{first}和{second}需查看另行确认的记录。',
        'wrong_subject': '资料存放于第10至20号柜。',
        'outside_scope': '（2042年版）',
    }[kind]
    assert answer(q, query, ['资料甲.md', '资料乙.md'], [distractor, facts], indices=[0]) is None


@pytest.mark.parametrize('domain,first,second,value_a,value_b', DOMAINS)
def test_complete_conditions_beat_shorter_partial_restatement(domain, first, second, value_a, value_b):
    q, query, _ = sample(domain, first, second, value_a, value_b)
    partial = f'{first}为{value_a}。'
    facts = f'一般条件下{first}为{value_a}；满足附加条件时{second}为{value_b}。'
    result = answer(q, query, ['资料乙.md', '资料甲.md'], [partial, facts])
    assert facts in result
    assert '来源：资料甲.md' in result
    assert '资料乙.md' not in result


@pytest.mark.parametrize('domain,first,second,value_a,value_b', DOMAINS)
def test_source_only_question_keeps_existing_nonquantity_lookup(domain, first, second, value_a, value_b):
    result = answer('2042年版资料在哪个文件？', '2042年版资料', ['资料甲.md'], ['（2042年版）'])
    assert '（2042年版）' in result


@pytest.mark.parametrize('domain,first,second,value_a,value_b', DOMAINS)
def test_complete_selected_numeric_evidence_stays_local(monkeypatch, tmp_path, capsys, domain, first, second, value_a, value_b):
    from app import chat_loop as runtime
    from app.dialog.state_machine import ConversationState
    from app.domain_host.host import EmptyDomainHost
    from public_tests.state.test_repo_meta_generic_file_list_scope_contract import (
        _EmbeddingStub, _RecordingLogger, _indexed_repo_state,
    )
    q, query, facts = sample(domain, first, second, value_a, value_b)
    repo = _indexed_repo_state(['资料甲.md', '资料乙.md'], [f'（2042年版）\n{facts}', '（2042年版）'])
    inputs = iter([q, 'exit'])
    monkeypatch.setattr(runtime, '_read_user_question', lambda **kw: next(inputs))
    monkeypatch.setattr(runtime, '_flush_pending_tty_input_unix', lambda: False)
    monkeypatch.setattr(runtime, 'conversation_state', ConversationState())
    monkeypatch.setattr('app.retrieval_flow.query.rewrite_search_query', lambda *a, **kw: query)

    def forbidden_remote(**kw):
        pytest.fail('Selected numeric evidence must be delivered locally')

    runtime.run_chat_loop(
        repo, _EmbeddingStub(), SimpleNamespace(models=SimpleNamespace(generate_content=forbidden_remote)),
        'offline-model', 'http://127.0.0.1:9', 'offline-model', _RecordingLogger(),
        notes_dir=tmp_path / 'notes', change_log_file=tmp_path / 'changes.db',
        domain_dispatch_port=EmptyDomainHost(),
    )
    output = capsys.readouterr().out
    assert facts in output
    assert '来源：资料甲.md' in output
    assert '[远程模型生成]' not in output
