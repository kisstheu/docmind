"""Synthetic domain-independent execution and bypass contracts; no live services."""
from copy import deepcopy
from decimal import Decimal
import json
from types import SimpleNamespace

import pytest

from ai.evidence_dependency import execute_delivery
from ai.evidence_dependency.executor import CONFIRMED, HYPOTHETICAL, UNKNOWN, Result, calculate
from ai.evidence_dependency.protocol import Proposal, build_prompt, needs_dependencies
from ai.evidence_dependency.semantic import verify_relations
from ai.decision_result import render_decision_result
from public_tests.state.test_evidence_scope_binding_contract import _candidate


def ref(c, start=1, end=None):
    end = end or len(c.text.splitlines())
    return dict(source_id=c.source_id, line_start=start, line_end=end,
                quote='\n'.join(c.text.splitlines()[start-1:end]))


def supported(checks, sources, question):
    return {c['id']: dict(status='SUPPORTED', basis='Synthetic controlled semantic oracle', evidence=c['refs'],
                         **({'comparison_operator': c['claim']['comparator']} if c['kind']=='condition_match' else {})) for c in checks}


def fixture(domain='商品', *, explicit=True, hypothetical=False, cross=False):
    labels = {'商品': ('商品甲', '60', '元', '12', '支'),
              '合同': ('合同甲', '1200', '元', '12', '个月'),
              '设备': ('设备甲', '900', 'GB', '3', '台')}
    subject, a, au, b, bu = labels[domain]
    texts = [f'{subject}本期总量{a}{au}，对应数量见条款R。', f'条款R：{subject}本期对应{b}{bu}。'] if cross else [
        f'{subject}本期总量{a}{au}，对应{b}{bu}。' if explicit else
        f'{subject}本期显示{a}{au}。\n历史记录：其他版本数量{b}{bu}。']
    cs = [_candidate(t, f'合成{domain}.md', i) for i, t in enumerate(texts)]
    fs = [dict(id='a', subject='obj', attribute='显示总量', value=a, unit=au, scope='本期', kind='source', requirement='none', refs=[ref(cs[0])]),
          dict(id='b', subject='obj', attribute='对应数量', value=b, unit=bu, scope='本期' if explicit else '历史其他版本', kind='source', requirement='none', refs=[ref(cs[-1])])]
    rs = [dict(id='r', inputs=['a','b'], claim=f'{a}{au}对应{b}{bu}', hypothetical=hypothetical, refs=[ref(c) for c in cs])]
    ds = [dict(id='d', subject='obj', label='平均量', op='divide', inputs=['a','b'], relation='r', unit=f'{au}/{bu}', comparator='eq', hypothetical=hypothetical)]
    return dict(version=1, objects=[dict(id='obj',label=subject)], facts=fs, relations=rs,
                derivations=ds, decisions=[], delivery=['a','b','d']), cs


def run(p, cs, verifier=supported, question='计算平均量。'):
    return execute_delivery(json.dumps(p, ensure_ascii=False), cs, question, verifier=verifier)


def source_fact(node_id, subject, attribute, value, scope, candidate):
    return dict(
        id=node_id,
        subject=subject,
        attribute=attribute,
        value=value,
        unit='',
        scope=scope,
        kind='source',
        requirement='none',
        refs=[ref(candidate)],
    )


def test_dependency_prompt_requires_every_linked_prerequisite_as_a_delivered_fact():
    candidate = _candidate(
        '只有在预算获批、法务复核、双方签署、保证金到账时，合同才允许生效。',
        '合成合同.md',
    )

    prompt = build_prompt('合同在什么条件下允许生效？', [candidate])

    assert '每个明确前提' in prompt
    assert '分别提取为source fact' in prompt
    assert '不得只保留类别、结论或其中一个前提' in prompt


def test_complete_contract_prerequisite_chain_survives_parse_execution_and_delivery():
    candidate = _candidate(
        '只有在预算获批、法务复核、双方签署、保证金到账时，合同才允许生效。',
        '合成合同.md',
    )
    values = ('预算获批', '法务复核', '双方签署', '保证金到账', '允许生效')
    facts = [
        source_fact(f'f{index}', 'contract', f'必要条件{index}', value,
                    '合同生效的共同条件', candidate)
        for index, value in enumerate(values, 1)
    ]
    proposal = dict(
        version=1,
        objects=[dict(id='contract', label='合同甲')],
        facts=facts,
        relations=[],
        derivations=[],
        decisions=[],
        delivery=[fact['id'] for fact in facts],
    )

    decision, execution = run(
        proposal, [candidate], question='合同在什么条件下允许生效？'
    )
    rendered = render_decision_result(decision)

    assert all(value in rendered for value in values)
    assert execution.diagnostics['row_nodes'] == [fact['id'] for fact in facts]
    assert decision.source_files == (candidate.path,)


def test_equal_durations_with_distinct_procurement_conditions_do_not_merge():
    arrival = _candidate('条件A：到货后24小时内完成验收。', '合成采购验收.md', 1)
    payment = _candidate('条件B：质检通过后24小时内完成付款。', '合成采购付款.md', 2)
    facts = [
        source_fact('arrival', 'arrival_rule', '时限', '24小时', '到货后完成验收', arrival),
        source_fact('payment', 'payment_rule', '时限', '24小时', '质检通过后完成付款', payment),
    ]
    proposal = dict(
        version=1,
        objects=[
            dict(id='arrival_rule', label='验收规则'),
            dict(id='payment_rule', label='付款规则'),
        ],
        facts=facts,
        relations=[],
        derivations=[],
        decisions=[],
        delivery=['arrival', 'payment'],
    )

    decision, execution = run(proposal, [arrival, payment], question='比较两项24小时要求。')
    rows = decision.comparison_table.rows

    assert len(rows) == 2
    assert ('验收规则', '24小时（到货后完成验收）', arrival.path) == (rows[0][0], rows[0][2], rows[0][-1])
    assert ('付款规则', '24小时（质检通过后完成付款）', payment.path) == (rows[1][0], rows[1][2], rows[1][-1])
    assert execution.diagnostics['row_nodes'] == ['arrival', 'payment']


def test_simple_course_fact_delivery_is_unchanged_without_prerequisites():
    candidate = _candidate('课程甲共8课时。', '合成课程制度.md')
    fact = source_fact('hours', 'course', '课时', '8课时', '当前课程', candidate)
    proposal = dict(
        version=1,
        objects=[dict(id='course', label='课程甲')],
        facts=[fact],
        relations=[],
        derivations=[],
        decisions=[],
        delivery=['hours'],
    )

    decision, execution = run(proposal, [candidate], question='课程甲有多少课时？')
    row = decision.comparison_table.rows[0]

    assert row[0:3] == ('课程甲', '课时', '8课时（当前课程）')
    assert row[-1] == candidate.path
    assert execution.diagnostics['row_nodes'] == ['hours']


@pytest.mark.parametrize('domain', ['商品', '合同', '设备'])
@pytest.mark.parametrize('cross', [False, True])
def test_explicit_and_cross_chunk_calculation_remains_useful(domain, cross):
    p, cs = fixture(domain, cross=cross)
    decision, ex = run(p, cs)
    assert ex.results['d'].state == CONFIRMED
    assert ex.results['d'].value == {'商品':5, '合同':100, '设备':300}[domain]
    assert cs[0].path in decision.source_files
    assert '平均量' in render_decision_result(decision)
    assert any(c['kind']=='operation_premise' for c in ex.checks)
    if cross: assert len(next(c for c in ex.checks if c['id']=='d')['refs']) >= 2


@pytest.mark.parametrize('domain', ['商品', '合同', '设备'])
@pytest.mark.parametrize('status', ['UNSUPPORTED','AMBIGUOUS','CONFLICT'])
def test_bad_relation_preserves_facts_not_derivation_or_exclusion(domain, status):
    p, cs = fixture(domain, explicit=False)
    def reject(checks, sources, q):
        result = supported(checks, sources, q)
        if 'd' in result: result['d']['status'] = status
        return result
    decision, ex = run(p, cs, reject)
    assert ex.results['a'].state == CONFIRMED
    assert ex.results['d'].value is None
    assert decision.selected_candidate is None and not decision.selected_source_files
    text = render_decision_result(decision)
    assert '明确不符合' not in text and '最低' not in text
    assert '待确认' in text and cs[0].path in text


@pytest.mark.parametrize('mutation', ['missing_relation','wrong_members','dangling','cycle','bad_quote','bad_lines','duplicate','self_verified','free_reason','hidden_derivation'])
def test_adversarial_proposals_cannot_bypass_required_contract(mutation):
    p, cs = fixture()
    if mutation=='missing_relation': p['relations']=[]
    elif mutation=='wrong_members': p['relations'][0]['inputs']=['a']
    elif mutation=='dangling': p['derivations'][0]['inputs'][0]='missing'
    elif mutation=='cycle':
        p['derivations'][0]['inputs'][0]='d'; p['relations'][0]['inputs'][0]='d'
    elif mutation=='bad_quote': p['facts'][0]['refs'][0]['quote']='伪造'
    elif mutation=='bad_lines': p['facts'][0]['refs'][0]['line_start']=True
    elif mutation=='duplicate': p['facts'][1]['id']='a'
    elif mutation=='self_verified': p['derivations'][0]['status']='VERIFIED'
    elif mutation=='free_reason': p['reason']='5元/支，全部满足，立即下单'
    elif mutation=='hidden_derivation':
        p['facts'].append(dict(p['facts'][0],id='fake',attribute='已确认平均值',value='5',unit='元/支'))
        p['delivery'].append('fake')
    try:
        decision, ex = run(p,cs)
    except ValueError:
        assert mutation in {'bad_lines','duplicate','self_verified','free_reason'}
        return
    if mutation=='hidden_derivation': assert ex.results['fake'].state == UNKNOWN
    else: assert ex.results['d'].state == UNKNOWN and ex.results['d'].value is None
    assert decision.selected_candidate is None


def test_semantics_checks_direct_fact_even_if_value_occurs_elsewhere():
    p, cs = fixture()
    p['facts'][0].update(attribute='单件价', value='12', unit='元/支')
    def oracle(checks, sources, q):
        assert any(c['id']=='a' and c['claim']['unit']=='元/支' for c in checks)
        result=supported(checks,sources,q);result['a']['status']='UNSUPPORTED';return result
    _, ex = run(p,cs,oracle)
    assert ex.results['a'].state == UNKNOWN and ex.results['d'].value is None


@pytest.mark.parametrize('domain', ['商品','合同','设备'])
def test_hypothetical_propagation_cannot_rank_or_select(domain):
    p, cs=fixture(domain,hypothetical=True)
    p['derivations'][0]['hypothetical']=False  # malicious omission must not upgrade relation
    p['decisions']=[dict(id='choice',op='minimum',subject='',inputs=['d'],scope=['obj'])]
    decision, ex=run(p,cs)
    assert ex.results['d'].state==HYPOTHETICAL
    assert ex.results['choice'].state==UNKNOWN
    assert not decision.selected_candidate
    row=next(row for row in decision.comparison_table.rows if row[1]=='平均量')
    assert row[2]=='—' and '若' in row[3] and '尚未证实' in row[3]


def test_one_invalid_relation_does_not_destroy_independent_result():
    p, cs=fixture()
    p['relations'].append(dict(p['relations'][0],id='r2'))
    p['derivations'].append(dict(p['derivations'][0],id='d2',relation='r2'))
    def oracle(checks,sources,q):
        v=supported(checks,sources,q);v['d']['status']='UNSUPPORTED';return v
    _, ex=run(p,cs,oracle)
    assert ex.results['d'].value is None and ex.results['d2'].value==5


@pytest.mark.parametrize('op,a,au,b,bu,out,expected', [
    ('ceil_divide','100','支','12','支/盒','盒',9),
    ('multiply','9','盒','12','支/盒','支',108),
    ('subtract','108','支','100','支','支',8),
    ('add','0.1','元','0.2','元','元',Decimal('0.3')),
    ('divide','1200','元','12','月','元/月',100),
    ('compare','10','GB','9','GB','',True),
])
def test_decimal_operations_and_integer_coverage(op,a,au,b,bu,out,expected):
    assert calculate(op,Result('a',value=a,unit=au),Result('b',value=b,unit=bu),out,'gt')==expected


@pytest.mark.parametrize('a,au,b,bu,out', [('NaN','元','1','支','元/支'),('inf','元','1','支','元/支'),
    ('1','元','0','支','元/支'),('12','元','12','支','元'),('1000','元','1','支','万元/支')])
def test_invalid_numbers_units_and_zero_are_rejected(a,au,b,bu,out):
    with pytest.raises(ValueError): calculate('divide',Result('a',value=a,unit=au),Result('b',value=b,unit=bu),out)


def test_local_semantic_failure_never_allows_critical_result():
    p,cs=fixture()
    class Session:
        trust_env=True
        def post(self,*args,**kwargs):
            assert not self.trust_env
            raise ValueError('malformed local response')
    def verifier(checks,sources,q): return verify_relations(checks,sources,q,session=Session())
    decision,ex=run(p,cs,verifier)
    assert ex.results['d'].state==UNKNOWN and not decision.selected_candidate
    assert '原文陈述' in render_decision_result(decision)


def test_ordinary_low_risk_does_not_request_dependency_graph():
    assert not needs_dependencies('设备甲是否支持离线运行？','normal_retrieval')
    assert needs_dependencies('比较这些方案并推荐一个。','normal_retrieval')


def matching_fixture():
    p,cs=fixture()
    question='必须对应12支；优先本期。'
    user_ref=dict(source_id='user',line_start=1,line_end=1,quote=question)
    p['facts'].extend([
        dict(id='req',subject='USER',attribute='对应数量',value='12',unit='支',scope='',kind='requirement',requirement='hard',refs=[user_ref]),
        dict(id='pref',subject='USER',attribute='时间',value='本期',unit='',scope='',kind='requirement',requirement='preference',refs=[user_ref])])
    p['relations'].append(dict(id='rm',inputs=['b','req'],claim='本期对应数量满足用户明确条件',hypothetical=False,refs=[ref(cs[0])]))
    p['derivations'].append(dict(id='m',subject='obj',label='数量条件',op='match',inputs=['b','req'],relation='rm',unit='',comparator='eq',hypothetical=False))
    p['decisions']=[dict(id='all',op='all_match',subject='obj',inputs=['m'],scope=['obj'])]
    return p,cs,question


@pytest.mark.parametrize('mutation', ['none','empty_requires','unknown','unasked','coverage_missing','explicit_mismatch'])
def test_hard_conditions_preferences_unknown_and_complete_match(mutation):
    p,cs,q=matching_fixture()
    def oracle(checks,sources,question):
        v=supported(checks,sources,question)
        if mutation=='unknown': v['m']['status']='AMBIGUOUS'
        if mutation=='coverage_missing': v['__requirements__']['status']='UNSUPPORTED'
        return v
    if mutation=='empty_requires': p['decisions'][0]['inputs']=[]
    if mutation=='unasked': p['facts'][2]['value']='用户未提出条件'
    if mutation=='explicit_mismatch':
        p['derivations'][1]['op']='mismatch'
        p['facts'][2]['value']='13';q=q.replace('12','13')
        for f in p['facts'][2:]:f['refs'][0]['quote']=q
    decision,ex=run(p,cs,oracle,q)
    if mutation=='none': assert ex.results['all'].state==CONFIRMED and ex.results['all'].value is True
    elif mutation=='explicit_mismatch': assert ex.results['all'].state==CONFIRMED and ex.results['all'].value is False
    else:
        assert ex.results['all'].state==UNKNOWN
        assert '明确不符合' not in render_decision_result(decision)
    assert decision.selected_candidate is None


def test_unknown_table_cannot_leak_unconditional_actions_or_selection():
    p,cs=fixture(explicit=False)
    def reject(checks,sources,q):
        v=supported(checks,sources,q);v['d']['status']='UNSUPPORTED';return v
    decision,ex=run(p,cs,reject)
    assert not decision.selected_source_files and not decision.selected_candidate
    unknown_row=next(row for row in decision.comparison_table.rows if row[1]=='平均量')
    assert unknown_row[2]=='—'
    assert unknown_row[5] in decision.next_actions
    assert all(row[-1] in decision.source_files for row in decision.comparison_table.rows)
    assert '下单' not in decision.next_actions and '最低' not in decision.conclusion


def test_real_runner_uses_one_proposal_zero_review_and_clears_stale_selection(monkeypatch,tmp_path):
    from app import chat_loop as runtime
    from app.dialog.state_machine import ConversationState
    from public_tests.state.test_repo_meta_generic_file_list_scope_contract import _run_turns
    from ai import evidence_dependency
    p,cs=fixture(explicit=False)
    calls=[]
    def generate(**kw):
        calls.append(kw)
        data=json.loads(kw['contents'].split('\n')[-1])
        source=data['evidence'][0]
        for group in ('facts','relations'):
            for item in p[group]:
                for reference in item['refs']: reference['source_id']=source['source_id']
        return SimpleNamespace(text=json.dumps(p,ensure_ascii=False))
    def reject(checks,sources,q,**kwargs):
        v=supported(checks,sources,q)
        if 'd' in v:v['d']['status']='UNSUPPORTED'
        return v
    monkeypatch.setattr(evidence_dependency,'verify_relations',reject)
    def forbidden(**kwargs):raise AssertionError('No remote review allowed')
    _run_turns(monkeypatch,tmp_path,questions=['比较商品甲并计算平均量。'],repo_paths=[cs[0].path],
        repo_chunks=[cs[0].text],state=ConversationState(last_selected_candidate='旧对象',last_selected_source_files=['旧资料.md']),
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
        evidence_reviewer=forbidden,delivery_strategy='dependency')
    state=runtime.conversation_state
    assert len(calls)==1
    assert state.last_selected_candidate is None and not state.last_selected_source_files
    assert state.last_answer_source_files==[cs[0].path]
    assert '60元' in state.last_answer_text and '待确认' in state.last_answer_text
    assert '5元/支' not in state.last_answer_text


def test_low_risk_runner_skips_new_semantics_and_remote_review(monkeypatch,tmp_path):
    from app import chat_loop as runtime
    from app.dialog.state_machine import ConversationState
    from public_tests.state.test_repo_meta_generic_file_list_scope_contract import _run_turns
    from ai import evidence_dependency
    calls=[]
    def generate(**kw):
        calls.append(kw);return SimpleNamespace(text='设备甲在本机运行，支持离线处理。来源：合成设备.md')
    def forbidden(*args,**kwargs):raise AssertionError('Unexpected semantic/review call')
    monkeypatch.setattr(evidence_dependency,'verify_relations',forbidden)
    _run_turns(monkeypatch,tmp_path,questions=['概括设备甲的工作方式。'],repo_paths=['合成设备.md'],repo_chunks=['设备甲支持离线运行。'],
        state=ConversationState(),client=SimpleNamespace(models=SimpleNamespace(generate_content=generate)),
        evidence_reviewer=forbidden,delivery_strategy='dependency')
    assert len(calls)==1
    assert '离线' in runtime.conversation_state.last_answer_text


def test_confirmed_comparable_values_can_rank_with_explicit_scope():
    p,cs=fixture()
    p['objects'].append(dict(id='other',label='商品乙'))
    other=_candidate('商品乙本期总量72元，对应12支。','合成乙.md')
    cs.append(other)
    p['facts'].extend([dict(p['facts'][0],id='a2',subject='other',value='72',refs=[ref(other)]),
                       dict(p['facts'][1],id='b2',subject='other',refs=[ref(other)])])
    p['relations'].append(dict(id='r2',inputs=['a2','b2'],claim='72元对应12支',hypothetical=False,refs=[ref(other)]))
    p['derivations'].append(dict(p['derivations'][0],id='d2',subject='other',inputs=['a2','b2'],relation='r2'))
    p['decisions']=[dict(id='rank',op='minimum',subject='',inputs=['d','d2'],scope=['obj','other'])]
    decision,ex=run(p,cs)
    assert ex.results['rank'].state==CONFIRMED and ex.results['rank'].value==('obj',)
    assert '在商品甲、商品乙之间' in decision.conclusion
    assert '最低' in decision.conclusion
    assert decision.selected_candidate is None  # ranking alone does not prove all user conditions


@pytest.mark.parametrize('reason', ['timeout','truncated','missing_check','bad_quote'])
def test_local_response_protocol_fails_closed(reason):
    p,cs=fixture()
    def verifier(checks,sources,q):
        class Session:
            def post(self,url,json,timeout):
                if reason=='timeout':
                    import requests
                    raise requests.Timeout('synthetic timeout')
                entries=[dict(id=c['id'],status='SUPPORTED',basis='synthetic',evidence=c['refs']) for c in checks]
                if reason=='missing_check': entries=entries[:-1]
                if reason=='bad_quote':
                    for e in entries:e['evidence']=[dict(e['evidence'][0],quote='伪造原文')]
                raw=dict(done=True,done_reason='length' if reason=='truncated' else 'stop',response=__import__('json').dumps(dict(checks=entries)))
                return SimpleNamespace(raise_for_status=lambda:None,json=lambda:raw)
        return verify_relations(checks,sources,q,session=Session())
    decision,ex=run(p,cs,verifier)
    assert ex.results['d'].value is None and not decision.selected_candidate


def test_direct_numeric_facts_can_be_compared_without_artificial_derivation():
    p,cs=fixture()
    other=_candidate('商品乙本期总量72元。','合成乙.md');cs.append(other)
    p['objects'].append(dict(id='other',label='商品乙'))
    p['facts'].append(dict(p['facts'][0],id='a2',subject='other',value='72',refs=[ref(other)]))
    p['relations']=[];p['derivations']=[]
    p['decisions']=[dict(id='rank',op='minimum',subject='',inputs=['a','a2'],scope=['obj','other'])]
    _,ex=run(p,cs)
    assert ex.results['rank'].state==CONFIRMED and ex.results['rank'].value==('obj',)


def test_conflicting_condition_results_cannot_be_overwritten_by_last_match():
    p,cs,q=matching_fixture()
    p['relations'].append(dict(p['relations'][-1],id='rm2'))
    p['derivations'].append(dict(p['derivations'][-1],id='m2',relation='rm2',op='mismatch'))
    p['decisions'][0]['inputs']=['m2','m']
    def inconsistent(checks,sources,question):
        v=supported(checks,sources,question);v['m2']['comparison_operator']='gt';return v
    decision,ex=run(p,cs,inconsistent,question=q)
    assert ex.results['all'].state=='CONFLICT'  and not decision.selected_candidate


def test_large_source_union_is_batched_without_dropping_independent_checks():
    cs=[_candidate(f'对象{i}的值为{i+1}。\n'+'完整范围限定。'*400,f'合成材料{i}.md') for i in range(6)]
    checks=[dict(id=str(i),kind='fact',claim=dict(kind='source'),refs=[ref(c,1,1)]) for i,c in enumerate(cs)]
    calls=[]
    class Session:
        def post(self,url,json,timeout):
            data=__import__('json').loads(json['prompt'].split('\n')[-1]);calls.append(data)
            entries=[dict(id=c['id'],status='SUPPORTED',basis='synthetic',evidence=c['refs']) for c in data['checks']]
            return SimpleNamespace(raise_for_status=lambda:None,json=lambda:dict(done=True,done_reason='stop',response=__import__('json').dumps(dict(checks=entries))))
    values=verify_relations(checks,cs,'列出原文事实。',session=Session())
    assert set(values)=={str(i) for i in range(6)} and len(calls)>1
    for call in calls:
        for evidence in call['evidence']:
            if evidence['source_id']=='user':continue
            original=next(c for c in cs if c.source_id==evidence['source_id'])
            assert '\n'.join(x['text'] for x in evidence['lines'])==original.text


def test_exact_unit_suffix_does_not_duplicate_display_or_prevent_arithmetic():
    p,cs=fixture()
    p['facts'][0]['value']='60元';p['facts'][1]['value']='12支'
    decision,ex=run(p,cs)
    assert ex.results['d'].value==5
    text=render_decision_result(decision)
    assert '元元' not in text and '支支' not in text


def test_omitted_user_condition_matches_remain_visible_unknown_actions():
    p,cs,q=matching_fixture()
    p['derivations']=[];p['decisions']=[]
    decision,ex=run(p,cs,question=q)
    assert '对应数量：12支' in decision.missing_information
    assert '核实商品甲是否满足这些用户条件' in decision.next_actions
    assert '明确不符合' not in decision.next_actions


@pytest.mark.parametrize('actual,operator,expected', [('12','ge',True),('11','ge',False),('11','le',True)])
def test_numeric_user_threshold_is_evaluated_by_code(actual,operator,expected):
    p,cs,q=matching_fixture()
    p['facts'][1]['value']=actual
    if actual=='11':
        cs=[_candidate(cs[0].text.replace('12','11'),cs[0].path)]
        for f in p['facts'][:2]:f['refs']=[ref(cs[0])]
        for r in p['relations']:r['refs']=[ref(cs[0])]
    def oracle(checks,sources,question):
        v=supported(checks,sources,question)
        for c in checks:
            if c['kind']=='condition_match':v[c['id']]['comparison_operator']=operator
        return v
    _,ex=run(p,cs,oracle,q)
    assert ex.results['m'].state==CONFIRMED and ex.results['m'].value is expected


def test_numeric_match_cannot_use_a_semantic_verdict_to_skip_local_comparison():
    p,cs,q=matching_fixture()
    def fake(checks,sources,question):
        v=supported(checks,sources,question);v['m']['comparison_operator']='semantic';return v
    _,ex=run(p,cs,fake,q)
    assert ex.results['m'].state==UNKNOWN and ex.results['all'].state==UNKNOWN
