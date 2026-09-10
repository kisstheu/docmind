"""Caller attestations are independent evidence; all fixtures are synthetic."""
from copy import deepcopy
from dataclasses import asdict
from decimal import Decimal
import json
import hashlib

import pytest

from ai.evidence_dependency import execute_delivery
from ai.evidence_dependency.executor import execute, UNKNOWN, CONFIRMED
from ai.evidence_dependency.protocol import Proposal
from ai.evidence_dependency.delivery import deliver
from ai.evidence_dependency.user_evidence import (
    snapshot, targets, apply_records, confirmation_execution, identities,
)
from public_tests.state.test_evidence_dependency_contract import fixture, supported


def saved(domain='商品', reject=('b', 'd')):
    p, cs = fixture(domain)
    def verify(checks, sources, question):
        result = supported(checks, sources, question)
        for key in reject:
            if key in result:
                result[key]['status'] = 'AMBIGUOUS'
        return result
    _, ex = execute_delivery(json.dumps(p), cs, '计算本任务平均量。', verifier=verify)
    return snapshot(ex, cs, '计算本任务平均量。'), ex, cs


def record(t, value=None, records=()):
    values = {r['target_key']: r for r in records}
    return dict(confirmation_id='user-' + t['node_id'], object_key=t['object_key'],
        target_key=t['target_key'], status='active', provenance='user',
        value=value if value is not None else 'confirmed' if t['kind'] == 'relation' else t['value'], unit=t['unit'],
        input_values=[[i['target_key'], values.get(i['target_key'], i)['value'], i['unit']] for i in t['inputs']])


@pytest.mark.parametrize('domain', ['商品', '合同', '设备'])
def test_user_fact_and_relation_release_only_their_dependency(domain):
    state, ex, cs = saved(domain)
    ts = {t['node_id']: t for t in targets(ex, cs)}
    fact = record(ts['b'])
    ex, _ = apply_records(state, [fact])
    assert ex.results['b'].state == CONFIRMED
    assert ex.results['d'].state == UNKNOWN
    relation = record(ts['d'], records=[fact])
    ex, cs = apply_records(state, [fact, relation])
    assert ex.results['d'].state == CONFIRMED
    assert ex.results['d'].value == {'商品': 5, '合同': 100, '设备': 300}[domain]
    decision = deliver(ex, cs, state['question'])
    assert '依据用户确认' in str(decision.comparison_table.rows)
    assert '用户已确认' in str(decision.comparison_table.rows)
    assert decision.selected_candidate is None and not decision.selected_source_files
    revoked, _ = apply_records(state, [])
    assert revoked.results['a'].state == CONFIRMED
    assert revoked.results['b'].state == UNKNOWN and revoked.results['d'].state == UNKNOWN


def test_user_supplies_value_absent_from_source_without_requiring_web_proof():
    state, ex, cs = saved()
    ts = {t['node_id']: t for t in targets(ex, cs)}
    fact = record(ts['b'], '20')
    relation = record(ts['d'], records=[fact])
    after, _ = apply_records(state, [fact, relation])
    assert after.results['d'].value == Decimal('3')
    assert state['proposal']['facts'][1]['value'] == '12'
    assert after.results['b'].user_confirmations == (fact['confirmation_id'],)


def test_correction_invalidates_previous_semantic_premise_but_keeps_independent_fact():
    state, ex, cs = saved(reject=())
    t = next(t for t in targets(ex, cs) if t['node_id'] == 'b')
    after, _ = apply_records(state, [record(t, '30')])
    assert after.results['a'].value == 60
    assert after.results['b'].value == 30
    assert after.results['d'].state == UNKNOWN


@pytest.mark.parametrize('mutation', ['object', 'target', 'unit', 'empty', 'nan', 'provenance', 'duplicate'])
def test_invalid_user_input_is_rejected(mutation):
    state, ex, cs = saved()
    t = next(t for t in targets(ex, cs) if t['node_id'] == 'b')
    r = record(t)
    if mutation == 'object': r['object_key'] = 'other'
    elif mutation == 'target': r['target_key'] = 'other'
    elif mutation == 'unit': r['unit'] = 'other'
    elif mutation == 'empty': r['value'] = ''
    elif mutation == 'nan': r['value'] = 'NaN'
    elif mutation == 'provenance': r['provenance'] = 'web'
    with pytest.raises(ValueError):
        apply_records(state, [r, r] if mutation == 'duplicate' else [r])


def test_changed_node_ids_rebind_and_changed_content_or_same_page_object_does_not():
    state, ex, cs = saved()
    before = {t['target_key'] for t in targets(ex, cs)}
    p = deepcopy(state['proposal'])
    for group in ('objects', 'facts', 'relations', 'derivations'):
        for n in p[group]:
            n['id'] = 'new_' + n['id']
            if 'subject' in n: n['subject'] = 'new_' + n['subject']
            if 'inputs' in n: n['inputs'] = ['new_' + x for x in n['inputs']]
            if 'relation' in n: n['relation'] = 'new_' + n['relation']
    p['delivery'] = ['new_' + x for x in p['delivery']]
    changed = execute(Proposal.model_validate(p), cs, state['question'], lambda *_: {})
    assert {t['target_key'] for t in targets(changed, cs)} == before
    p['objects'][0]['label'] = '同页另一对象'
    changed = execute(Proposal.model_validate(p), cs, state['question'], lambda *_: {})
    assert not ({t['target_key'] for t in targets(changed, cs)} & before)
    other = tuple(type('Source', (), dict(source_id=c.source_id, path=c.path, text=c.text + '\n来源新版本'))() for c in cs)
    changed = execute(ex.proposal, other, state['question'], lambda *_: {})
    assert not ({t['target_key'] for t in targets(changed, other)} & before)


def test_model_or_web_cannot_create_user_authority():
    state, ex, cs = saved()
    p = deepcopy(state['proposal'])
    p['facts'][1]['kind'] = 'USER_CONFIRMED'
    with pytest.raises(ValueError):
        Proposal.model_validate(p)
    with pytest.raises(ValueError):
        execute(ex.proposal, cs, state['question'], lambda *_: {}, user_evidence={'facts': {'b': '12'}})
    ex = execute(ex.proposal, cs, state['question'], lambda *_: {'b': {'status': 'USER_CONFIRMED'}})
    assert ex.results['b'].state == UNKNOWN


def test_generic_missing_device_attribute_is_user_input_and_other_gap_stays_unknown():
    state, ex, cs = saved('设备')
    q = '设备型号为M01。\n供货数量为20台。'
    p = state['proposal']
    p['facts'].extend([dict(id=key, subject='USER', attribute=label, value=value, unit=unit,
        kind='requirement', requirement='hard', scope='当前条件',
        refs=[dict(source_id='user', line_start=line, line_end=line, quote=q.splitlines()[line-1])])
        for key,label,value,unit,line in [('model','型号','M01','',1),('supply','供货数量','20','台',2)]])
    before = execute(Proposal.model_validate(p), cs, q, supported)
    augmented, cs = confirmation_execution(snapshot(before, cs, q))
    state = snapshot(augmented, cs, q)
    ts = targets(augmented, cs)
    f = next(t for t in ts if t['label'] == '型号' and t['kind'] == 'fact')
    r = next(t for t in ts if t['label'] == '型号条件匹配')
    fact = record(f, 'M01')
    relation = record(r, records=[fact])
    after, _ = apply_records(state, [fact, relation])
    assert after.results[f['node_id']].value == 'M01'
    assert after.results[r['node_id']].value is True
    remaining = next(t for t in ts if t['label'] == '供货数量条件匹配')
    assert after.results[remaining['node_id']].state == UNKNOWN


def test_independent_calculation_survives_confirmation_and_revocation():
    p, cs = fixture()
    other, other_cs = fixture('合同')
    for group in ('objects', 'facts', 'relations', 'derivations'):
        for n in other[group]:
            n['id'] = 'independent_' + n['id']
            if 'subject' in n: n['subject'] = 'independent_' + n['subject']
            if 'inputs' in n: n['inputs'] = ['independent_' + x for x in n['inputs']]
            if 'relation' in n: n['relation'] = 'independent_' + n['relation']
        p[group].extend(other[group])
    cs.extend(other_cs)
    def verify(checks, sources, question):
        verdicts = supported(checks, sources, question)
        verdicts['d']['status'] = 'AMBIGUOUS'
        return verdicts
    ex = execute(Proposal.model_validate(p), cs, '计算', verify)
    state = snapshot(ex, cs, '计算')
    t = next(t for t in targets(ex, cs) if t['node_id'] == 'd')
    for records in ([record(t)], []):
        after, _ = apply_records(state, records)
        assert after.results['independent_d'].state == CONFIRMED
        assert after.results['independent_d'].value == 100


def test_new_user_value_conflicting_with_same_scoped_fact_is_not_silently_ranked():
    state, _, cs = saved(reject=())
    p = state['proposal']
    second = {**p['facts'][1], 'id': 'other_fact', 'refs': [dict(p['facts'][1]['refs'][0], quote='12')]}
    p['facts'].append(second)
    ex = execute(Proposal.model_validate(p), cs, state['question'], supported)
    state = snapshot(ex, cs, state['question'])
    t = next(t for t in targets(ex, cs) if t['node_id'] == 'b')
    after, _ = apply_records(state, [record(t, '30')])
    assert after.results['b'].state == 'CONFLICT'
    assert after.results['other_fact'].state == 'CONFLICT'
    assert after.results['d'].state != CONFIRMED


def test_confirmed_hypothetical_relation_does_not_leave_known_inputs_hypothetical():
    p, cs = fixture(hypothetical=True)
    ex = execute(Proposal.model_validate(p), cs, '计算', supported)
    state = snapshot(ex, cs, '计算')
    t = next(t for t in targets(ex, cs) if t['node_id'] == 'd')
    after, _ = apply_records(state, [record(t)])
    assert after.results['d'].state == CONFIRMED


def test_facade_checks_saved_file_bytes_and_separate_user_record_file(tmp_path):
    from app.request_facade import replay
    state, ex, cs = saved()
    for c in cs:
        (tmp_path/c.path).write_text(c.text)
    state['source_manifest'] = {c.path: hashlib.sha256((tmp_path/c.path).read_bytes()).hexdigest() for c in cs}
    t = next(t for t in targets(ex, cs) if t['node_id'] == 'b')
    user_file = tmp_path/'attestations.json'
    user_file.write_text(json.dumps([record(t)]))
    data = {'execution_snapshot': state}
    result = replay(data, user_file, tmp_path)
    assert result['model_calls'] == {'remote_generation': 0, 'local_verification': 0}
    assert '用户已确认' in result['answer']
    (tmp_path/cs[0].path).write_text('另一证据版本')
    with pytest.raises(ValueError, match='source version changed'):
        replay(data, user_file, tmp_path)


def test_changed_transitive_input_requires_review_only_for_downstream_attestation():
    p, cs = fixture()
    p['relations'].append(dict(id='total_relation',inputs=['d','b'],claim='平均量与数量对应本期总量',hypothetical=False,refs=p['relations'][0]['refs']))
    p['derivations'].append(dict(id='total',subject='obj',label='本期总量',op='multiply',inputs=['d','b'],relation='total_relation',unit='元',comparator='eq',hypothetical=False))
    def verify(checks,sources,question):
        verdicts=supported(checks,sources,question)
        for key in ('d','total'):verdicts[key]['status']='AMBIGUOUS'
        return verdicts
    ex=execute(Proposal.model_validate(p),cs,'计算',verify)
    state=snapshot(ex,cs,'计算')
    ts={t['node_id']:t for t in targets(ex,cs)}
    after,_=apply_records(state,[record(ts['d']),record(ts['total'])])
    assert after.results['d'].value==5 and after.results['d'].state==CONFIRMED
    assert after.results['total'].state==UNKNOWN
    assert after.diagnostics['confirmation_needs_review']==['user-total']


def test_numeric_user_condition_requires_explicit_direction_instead_of_assuming_equality():
    p,cs=fixture('设备')
    q='供货数量至少20台。'
    p['facts'].append(dict(id='required_quantity',subject='USER',attribute='供货数量',value='20',unit='台',scope='本任务',kind='requirement',requirement='hard',refs=[dict(source_id='user',line_start=1,line_end=1,quote=q)]))
    before=execute(Proposal.model_validate(p),cs,q,supported)
    ex,cs=confirmation_execution(snapshot(before,cs,q));state=snapshot(ex,cs,q)
    ts=targets(ex,cs)
    fact=record(next(t for t in ts if t['label']=='供货数量'),'30')
    target=next(t for t in ts if t['label']=='供货数量条件匹配')
    relation=record(target,records=[fact])
    assert target['requires_comparator']
    with pytest.raises(ValueError,match='comparison direction'):
        apply_records(state,[fact,relation])
    relation['comparison_operator']='ge'
    after,_=apply_records(state,[fact,relation])
    assert after.results[target['node_id']].state==CONFIRMED
    assert after.results[target['node_id']].value is True
