"""Polar fact lookup must retain source paragraphs and reject topical headings."""
from types import SimpleNamespace

import pytest

from app.chat_text.lookup_answer_main import maybe_build_direct_lookup_answer


DOMAINS = [('采购', '收货', '抽检'), ('合同', '签署', '复核'), ('课程', '开课', '预习')]


def sample(domain, when, action):
    question = f'根据2042年版{domain}说明，{when}前是否推荐常规{action}？仅在哪两类情况下需要{action}？'
    query = f'2042 年版 {domain}说明 {action} 需要{action}'
    facts = (
        f'不推荐在{when}前常规{action}，仅以下情况需要{action}：'
        f'①已有明确异常记录。此类情况如确有必要继续{when}，须先核对原始记录、'
        '确认负责人员与应急条件，取得相关方同意后，选择与既有问题无关联的替代方案，'
        f'再进行{action}，结果仅用于本次核验；'
        f'②双方约定明确要求{action}时，按约定执行。'
        '同时应进一步了解约定的适用范围、实施步骤、记录保存要求与后续处理流程，'
        '保留完整核验依据，不得将例外情况扩大为所有情况均需执行。'
    )
    assert len(facts) > 160
    return question, query, facts


def answer(question, query, paths, texts, indices=None, **kwargs):
    return maybe_build_direct_lookup_answer(
        question=question, search_query=query,
        repo_state=SimpleNamespace(chunk_paths=paths, chunk_texts=texts),
        relevant_indices=list(range(len(texts))) if indices is None else indices,
        **kwargs,
    )


@pytest.mark.parametrize('domain,when,action', DOMAINS)
@pytest.mark.parametrize('order', [[0, 1, 2], [2, 1, 0]])
def test_later_complete_paragraph_beats_heading_and_cross_document_noise(domain, when, action, order):
    q, query, facts = sample(domain, when, action)
    result = answer(q, query, ['资料甲.md', '资料乙.md', '资料甲.md'],
                    ['（2042年版）', '需根据资料进行目录整理', facts], indices=order)
    assert facts in result
    assert '来源：资料甲.md' in result
    assert '资料乙.md' not in result
    assert '（2042年版）' not in result


@pytest.mark.parametrize('domain,when,action', DOMAINS)
def test_shorter_restatement_does_not_displace_complete_conditions(domain, when, action):
    q, query, facts = sample(domain, when, action)
    result = answer(q, query, ['资料甲.md'] * 3,
                    [f'{action}。', f'常规{action}无需执行，例外参见适用范围。', facts])
    assert facts in result


@pytest.mark.parametrize('domain,when,action', DOMAINS)
def test_source_diversity_does_not_add_weaker_topic_only_paragraph(domain, when, action):
    q, query, facts = sample(domain, when, action)
    distractor = f'本册展示了{action}，相关表格见目录。'
    result = answer(q, query, ['资料乙.md', '资料甲.md'], [distractor, facts])
    assert facts in result
    assert '资料乙.md' not in result
    assert distractor not in result


@pytest.mark.parametrize('domain,when,action', DOMAINS)
@pytest.mark.parametrize('kind', ['title_only', 'irrelevant', 'empty', 'wrong_subject'])
def test_insufficient_factual_evidence_is_not_a_local_answer(domain, when, action, kind):
    q, query, _ = sample(domain, when, action)
    text = {'title_only': f'（2042年版）\n{action}指导原则\n（二）{action}。',
            'irrelevant': '需根据资料进行目录整理。', 'empty': '',
            'wrong_subject': '不推荐常规归档，仅在出现异常时处理。'}[kind]
    assert answer(q, query, ['资料甲.md'], [text]) is None


@pytest.mark.parametrize('domain,when,action', DOMAINS)
def test_complete_short_factual_answer_remains_local(domain, when, action):
    facts = f'不推荐常规{action}。'
    assert facts in answer(f'是否推荐常规{action}？', action, ['资料甲.md'], [facts], force_local_evidence=True)


@pytest.mark.parametrize('domain,when,action', DOMAINS)
def test_outside_selected_scope_cannot_supply_evidence(domain, when, action):
    q, query, facts = sample(domain, when, action)
    assert answer(q, query, ['资料甲.md', '资料乙.md'], ['（2042年版）', facts], indices=[0]) is None


@pytest.mark.parametrize('domain,when,action', DOMAINS)
def test_complete_evidence_uses_real_chat_loop_without_remote(monkeypatch, tmp_path, capsys, domain, when, action):
    from app import chat_loop as runtime
    from app.dialog.state_machine import ConversationState
    from app.domain_host.host import EmptyDomainHost
    from public_tests.state.test_repo_meta_generic_file_list_scope_contract import (
        _EmbeddingStub, _RecordingLogger, _indexed_repo_state,
    )
    q, query, facts = sample(domain, when, action)
    repo = _indexed_repo_state(['资料甲.md', '资料乙.md'], [f'（2042年版）\n{facts}', '需根据资料进行目录整理'])
    inputs = iter([q, 'exit'])
    monkeypatch.setattr(runtime, '_read_user_question', lambda **kw: next(inputs))
    monkeypatch.setattr(runtime, '_flush_pending_tty_input_unix', lambda: False)
    monkeypatch.setattr(runtime, 'conversation_state', ConversationState())
    monkeypatch.setattr('app.retrieval_flow.query.rewrite_search_query', lambda *a, **kw: query)
    def forbidden_remote(**kw):
        pytest.fail('Complete selected local evidence must not invoke remote generation')
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
