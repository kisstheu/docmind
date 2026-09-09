"""Nonliteral labels need a bound value proposition in selected evidence."""
from types import SimpleNamespace

import pytest

from public_tests.retrieval.test_direct_factual_evidence_contract import answer


CASES = [
    (
        '合同甲', '签署日期、合同期限和违约金额',
        '双方于2042年7月9日签署，自次日起生效，约定持续18个月；违约时支付80元。',
    ),
    (
        '课程甲', '开始时间、持续时长和及格标准',
        '上午9点开始，连续进行45分钟；得分达到72分时判为及格。',
    ),
    (
        '订单甲', '交付时间、包装件数和合格阈值',
        '确认后8天交付，每箱装24件；若完好比例≥96%则认定为合格。',
    ),
]


def sample(topic, labels, facts):
    question = f'按2042年版执行说明，{topic}的{labels}分别是多少？'
    query = f'2042 执行说明 {topic} ' + labels.replace('、', ' ').replace('和', ' ')
    # Scope comes from the actual selected chunk, never its filename.
    return question, query, f'{topic}\n{facts}'


@pytest.mark.parametrize('topic,labels,facts', CASES)
@pytest.mark.parametrize('order', [[0, 1], [1, 0]])
def test_bound_nonliteral_proposition_forms_local_evidence(topic, labels, facts, order):
    q, query, text = sample(topic, labels, facts)
    result = answer(q, query, ['资料甲.md', '资料乙.md'], [text, '（2042年版）'], indices=order)
    assert result is not None
    assert facts in result
    assert '来源：资料甲.md' in result
    assert '资料乙.md' not in result


@pytest.mark.parametrize('topic,labels,facts', CASES)
@pytest.mark.parametrize('kind', ['metadata', 'disconnected', 'wrong_chunk', 'outside_scope', 'wrong_inline_topic'])
def test_numbers_without_a_bound_requested_proposition_fail_closed(topic, labels, facts, kind):
    q, query, text = sample(topic, labels, facts)
    noise = {
        'metadata': f'{topic}\n2042年，第18页，编号80，版本2.4。',
        'disconnected': f'{topic}\n签署说明另行查看。第2042页7号档案9次修订；及格与合格要求另查。总数为72。',
        'wrong_chunk': '另一个主题\n' + facts,
        'outside_scope': '（2042年版）',
        'wrong_inline_topic': f'{topic}\n另一个主题：{facts}',
    }[kind]
    assert answer(q, query, ['资料甲.md', '资料乙.md'], [noise, text], indices=[0]) is None


@pytest.mark.parametrize('topic,labels,facts', CASES)
def test_scope_in_another_chunk_or_filename_is_not_authority(topic, labels, facts):
    q, query, _ = sample(topic, labels, facts)
    assert answer(q, query, [f'{topic}.md'] * 2, [topic, facts]) is None


@pytest.mark.parametrize('topic,labels,facts', CASES)
def test_uninterpretable_attribute_does_not_borrow_other_values(topic, labels, facts):
    question = f'{topic}的归档编号和复核标识分别是多少？'
    assert answer(question, f'{topic} 归档编号 复核标识', ['资料甲.md'], [f'{topic}\n{facts}']) is None


@pytest.mark.parametrize('facts', [
    '得分达到72分时判为不及格。',
    '得分达到72分时不应判为及格。',
    '若得分达到72分时不应判为及格。',
    '得分达到72分时是否判为及格？',
    '得分达到72分。另一个记录判为及格。',
    '第72页记载了及格要求。',
    '得分72分，及格情况另行核实。',
    '假设得分达到72分时判为及格。',
    '达到72分时判为及格？请另行核实。',
])
def test_outcome_must_be_affirmed_by_the_same_quantitative_condition(facts):
    assert answer('课程甲的及格标准是多少？', '课程甲 及格标准', ['资料甲.md'],
                  [f'课程甲\n{facts}'], force_local_evidence=True) is None


@pytest.mark.parametrize('facts', [
    '若双方于2042年7月9日签署，则另行确认期限。',
    '双方计划于2042年7月9日签署。',
    '双方并未于2042年7月9日签署。',
    '签署条目在2042年版第7页第9行。',
    '双方签署后于2042年7月9日归档。',
    '于2042年7月9日签署？请另行核实。',
])
def test_calendar_date_must_bind_to_the_requested_event(facts):
    assert answer('合同甲的签署日期是多少？', '合同甲 签署日期', ['资料甲.md'],
                  [f'合同甲\n{facts}'], force_local_evidence=True) is None


@pytest.mark.parametrize('topic,labels,facts', CASES)
def test_existing_single_value_and_exact_labels_keep_short_answers(topic, labels, facts):
    fact = '执行次数为4次。'
    result = answer(f'{topic}的执行次数是多少？', '执行次数', ['资料甲.md'],
                    [f'{topic}\n{fact}\n{facts}'], force_local_evidence=True)
    assert fact in result and facts not in result
    assert len(result) < 160


@pytest.mark.parametrize('topic,labels,facts', CASES)
def test_nonliteral_mapping_can_be_disabled_without_changing_exact_lookup(monkeypatch, topic, labels, facts):
    monkeypatch.setenv('DOCMIND_NONLITERAL_QUANTITY_FACTS', '0')
    q, query, text = sample(topic, labels, facts)
    assert answer(q, query, ['资料甲.md'], [text]) is None
    fact = '执行次数为4次。'
    assert fact in answer('执行次数是多少？', '执行次数', ['资料甲.md'], [fact], force_local_evidence=True)


@pytest.mark.parametrize('topic,event,outcome', [
    ('合同甲', '签署', '生效'), ('课程甲', '开课', '通过'), ('订单甲', '交付', '接收'),
])
@pytest.mark.parametrize('kind', ['calendar', 'condition'])
def test_event_and_outcome_anchors_come_from_each_question(topic, event, outcome, kind):
    label, fact = (f'{event}日期', f'于2043年8月6日{event}。') if kind == 'calendar' else (
        f'{outcome}阈值', f'若完成比例≥88%则认定为{outcome}。',
    )
    result = answer(f'{topic}的{label}是多少？', f'{topic} {label}', ['资料甲.md'],
                    [f'{topic}\n{fact}'], force_local_evidence=True)
    assert fact in result


@pytest.mark.parametrize('topic,outcome', [('合同甲', '生效'), ('课程甲', '通过'), ('订单甲', '接收')])
def test_separate_paragraph_becomes_ranked_without_merging_per_file_selection(topic, outcome):
    import sys

    first = f'{topic}启用时，起始值为4单位。'
    later = f'若当前值较之前增加≥6单位则认定为{outcome}，附加记录仅作支持依据。'
    captured = []
    previous_profile = sys.getprofile()

    def capture(frame, event, arg):
        if event == 'return' and frame.f_code.co_name == '_build_direct_lookup_evidence_items':
            captured.extend(frame.f_locals['ranked'])

    sys.setprofile(capture)
    try:
        result = answer(f'{topic}的起始值和{outcome}阈值分别是多少？', f'{topic} 起始值 {outcome}阈值',
                        ['资料甲.md'], [f'{first}\n\n（二）结果确认。\n\n{later}'])
    finally:
        sys.setprofile(previous_profile)
    assert {item['line'] for item in captured} == {first, later}
    assert all(item['path'] == '资料甲.md' for item in captured)
    assert result.count('来源：资料甲.md') == 1
    assert (first in result) != (later in result)


@pytest.mark.parametrize('topic,labels,facts', CASES)
def test_complete_nonliteral_evidence_uses_chat_loop_without_remote(monkeypatch, tmp_path, capsys, topic, labels, facts):
    from app import chat_loop as runtime
    from app.dialog.state_machine import ConversationState
    from app.domain_host.host import EmptyDomainHost
    from public_tests.state.test_repo_meta_generic_file_list_scope_contract import (
        _EmbeddingStub, _RecordingLogger, _indexed_repo_state,
    )

    # This contract starts in the existing evidence route. The broader
    # "standard" phrasing also has a synthesis route, tested above at extraction.
    q, query, text = sample(topic, labels.replace('标准', '阈值'), facts)
    repo = _indexed_repo_state(['资料甲.md'], [text])
    inputs = iter([q, 'exit'])
    monkeypatch.setattr(runtime, '_read_user_question', lambda **kw: next(inputs))
    monkeypatch.setattr(runtime, '_flush_pending_tty_input_unix', lambda: False)
    monkeypatch.setattr(runtime, 'conversation_state', ConversationState())
    monkeypatch.setattr('app.retrieval_flow.query.rewrite_search_query', lambda *a, **kw: query)

    def forbidden_remote(**kw):
        pytest.fail('A bound local value proposition must not invoke remote generation')

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
