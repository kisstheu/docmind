"""Task-owned user evidence and replay. Model proposals cannot create this input.

The facade accepts records only through its separate caller-owned input file.
Snapshots and semantic verdicts are private execution data, never browser input.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import SimpleNamespace

from .protocol import Proposal, Fact, Relation, Derivation


def add_requirement_targets(proposal):
    """Materialize missing condition inputs as UNKNOWN, never invent an actual value."""
    p = proposal.model_copy(deep=True)
    for obj in p.objects:
        existing = {d.inputs[1] for d in p.derivations if d.subject == obj.id and d.op in {'match', 'mismatch'}}
        for requirement in list(p.facts):
            if requirement.kind != 'requirement' or requirement.requirement != 'hard' or requirement.id in existing:
                continue
            if len(p.facts) >= 64 or len(p.relations) >= 32 or len(p.derivations) >= 32:
                break
            key = digest([obj.id, requirement.id])[:20]
            fid, rid, did = 'user_fact_' + key, 'user_relation_' + key, 'user_match_' + key
            if {fid, rid, did} & {n.id for n in [*p.facts, *p.relations, *p.derivations, *p.decisions]}:
                continue
            p.facts.append(Fact(id=fid, subject=obj.id, attribute=requirement.attribute,
                value='', unit=requirement.unit, scope='本任务条件核实：' + requirement.scope[:140],
                kind='source', requirement='none', refs=[]))
            p.relations.append(Relation(id=rid, inputs=[fid, requirement.id], hypothetical=False, refs=[],
                claim=f'核实{obj.label}的{requirement.attribute}实际值，并确认它与当前条件「{requirement.value}{requirement.unit}」按相同含义、单位和适用范围比较。'[:400]))
            p.derivations.append(Derivation(id=did, subject=obj.id, label=(requirement.attribute + '条件匹配')[:60],
                op='match', inputs=[fid, requirement.id], relation=rid, unit='', comparator='eq', hypothetical=False))
    return p


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class UserEvidence:
    facts: dict
    operations: dict


def identities(proposal, candidates):
    sources = {c.source_id: c for c in candidates}
    def refs(items):
        return sorted((sources[r.source_id].path, digest(sources[r.source_id].text), r.quote)
                      for r in items if r.source_id in sources)
    objects = {o.id: digest([o.label, sorted({tuple(x) for f in proposal.facts
                if f.subject == o.id for x in refs(f.refs)})]) for o in proposal.objects}
    keys = {f.id: digest([objects.get(f.subject, f.subject), f.attribute, f.scope,
                         f.kind, refs(f.refs)]) for f in proposal.facts}
    for _ in proposal.derivations:
        for d in proposal.derivations:
            if all(i in keys for i in d.inputs):
                keys[d.id] = digest([objects.get(d.subject), d.op, d.label, d.unit,
                                     d.comparator, [keys[i] for i in d.inputs]])
    return objects, keys


def targets(execution, candidates):
    p = execution.proposal
    objects, keys = identities(p, candidates)
    names = {o.id: o.label for o in p.objects}
    facts = {f.id: f for f in p.facts}
    relations = {r.id: r for r in p.relations}
    sources = {c.source_id: c for c in candidates}
    out = []
    for n in [*p.facts, *p.derivations]:
        if n.subject not in names or n.id not in keys:
            continue
        is_fact = n.id in facts
        r = execution.results[n.id]
        if is_fact and facts[n.id].kind != "source":
            continue
        node_refs = n.refs if is_fact else relations[n.relation].refs
        if not node_refs:
            node_refs = [ref for f in p.facts if f.subject == n.subject for ref in f.refs]
        evidence = [dict(path=sources[ref.source_id].path, quote=ref.quote,
                         chunk_fingerprint=digest(sources[ref.source_id].text))
                    for ref in node_refs if ref.source_id in sources]
        if not evidence:
            continue
        item = dict(target_key=keys[n.id], object_key=objects[n.subject],
                    object_label=names[n.subject], node_id=n.id,
                    kind="fact" if is_fact else "relation", state=r.state,
                    label=n.attribute if is_fact else n.label,
                    question=(f"{n.attribute}的实际值是什么？" if is_fact else relations[n.relation].claim),
                    value=str(r.value) if r.value is not None else n.value if is_fact else "",
                    unit=n.unit if is_fact else "", scope=n.scope if is_fact else "本任务所列输入之间的对应关系",
                    evidence=evidence, inputs=[])
        if not is_fact:
            for key in n.inputs:
                v, f = execution.results[key], facts.get(key)
                item['inputs'].append(dict(target_key=keys[key], label=f.attribute if f else v.label,
                    object_label=names.get(v.subject, '当前用户条件'),
                    value=str(v.value) if v.value is not None else f.value if f else '',
                    unit=f.unit if f else v.unit, state=v.state, editable=bool(f and f.kind == 'source')))
            if n.op in {'match', 'mismatch'} and facts.get(n.inputs[1]) and facts[n.inputs[1]].unit:
                from .executor import number
                requirement = facts[n.inputs[1]]
                try:
                    number(requirement.value.removesuffix(requirement.unit))
                except ValueError:
                    pass
                else:
                    item['requires_comparator'] = True
        out.append(item)
    # Duplicate semantic identities require user review, never guess a row.
    counts = {t['target_key']: sum(x['target_key'] == t['target_key'] for x in out) for t in out}
    return [t for t in out if counts[t['target_key']] == 1]


def snapshot(execution, candidates, question):
    return dict(version=1, proposal=execution.proposal.model_dump(), question=question,
                candidates=[dict(source_id=c.source_id, path=c.path, text=c.text) for c in candidates],
                checks=execution.checks, verdicts=execution.verdicts)


def confirmation_execution(state):
    from .executor import execute
    p = add_requirement_targets(Proposal.model_validate(state['proposal']))
    cs = tuple(SimpleNamespace(**c) for c in state['candidates'])
    known = {c['id']: c for c in state['checks']}
    ex = execute(p, cs, state['question'], lambda checks, *_: {
        c['id']: state['verdicts'].get(c['id'], {}) for c in checks if c == known.get(c['id'])})
    return ex, cs


def apply_records(state, records):
    """Validate independently submitted records against the saved execution scope."""
    from .executor import execute, number
    p = Proposal.model_validate(state['proposal'])
    cs = tuple(SimpleNamespace(**c) for c in state['candidates'])
    old_checks = {c['id']: c for c in state['checks']}
    def cached(checks, _sources, _question):
        return {c['id']: state['verdicts'].get(c['id'], {}) for c in checks
                if c == old_checks.get(c['id'])}
    original = execute(p, cs, state['question'], cached)
    catalog = {t['target_key']: t for t in targets(original, cs)}
    fact_nodes = {f.id: f for f in p.facts}
    facts, operations, seen = {}, {}, set()
    for record in records:
        if record.get('status') != 'active':
            continue
        t = catalog.get(record.get('target_key'))
        if (not t or record.get('object_key') != t['object_key']
                or not record.get('confirmation_id') or record.get('provenance') != 'user'
                or record['target_key'] in seen):
            raise ValueError('user evidence target is ambiguous or invalid')
        seen.add(record['target_key'])
        value, unit = record.get('value'), record.get('unit', '')
        if t['kind'] == 'fact':
            if not isinstance(value, str) or not value.strip() or len(value) > 240:
                raise ValueError('user fact requires an explicit value')
            if not isinstance(unit, str) or unit != t['unit']:
                raise ValueError('user fact unit does not match target')
            if unit:
                # Declared numeric facts cannot receive an incomplete free-text quantity.
                original_value = fact_nodes[t['node_id']].value.removesuffix(unit)
                try:
                    number(original_value or '0')
                except ValueError:
                    pass
                else:
                    number(value.removesuffix(unit))
            facts[t['node_id']] = dict(value=value.strip(), unit=unit, record=record,
                                      sources=tuple(dict.fromkeys(e['path'] for e in t['evidence'])))
        else:
            if value != 'confirmed':
                raise ValueError('relation requires an explicit confirmation')
            if t.get('requires_comparator') and record.get('comparison_operator') not in {'eq', 'lt', 'le', 'gt', 'ge'}:
                raise ValueError('numeric requirement needs an explicit comparison direction')
            operations[t['node_id']] = record
    # A relationship attestation is limited to the input values explicitly displayed.
    # Re-evaluate transitive inputs before applying it; stale downstream attestations
    # require review while independent facts and computations remain available.
    _, keys = identities(p, cs)
    review = []
    while True:
        ex = execute(p, cs, state['question'], cached, user_evidence=UserEvidence(facts, operations))
        invalid = []
        for node_id, record in operations.items():
            d = next(d for d in p.derivations if d.id == node_id)
            expected = []
            for key in d.inputs:
                f, value = fact_nodes.get(key), ex.results[key]
                supplied = facts.get(key, {})
                expected.append([keys[key], supplied.get('value', str(value.value) if value.value is not None else f.value if f else ''),
                                 supplied.get('unit', value.unit)])
            if record.get('input_values') != expected:
                invalid.append(node_id)
        if not invalid:
            break
        for key in invalid:
            review.append(operations.pop(key)['confirmation_id'])
    ex.diagnostics['confirmation_needs_review'] = review
    return ex, cs


def presentation(execution, candidates):
    catalog = targets(execution, candidates)
    by_node = {t['node_id']: t for t in catalog}
    rows = []
    for key in execution.diagnostics.get('row_nodes', []):
        item = by_node.get(key)
        if item:
            row = [item['target_key']]
            row.extend(i['target_key'] for i in item['inputs'] if i['editable'] and i['state'] != 'CONFIRMED')
        elif key.startswith('gap:'):
            obj = key[4:]
            ids = {f.id for f in execution.proposal.facts if f.subject == obj}
            ids.update(d.id for d in execution.proposal.derivations if d.subject == obj)
            row = [t['target_key'] for t in catalog if t['node_id'] in ids and t['state'] != 'CONFIRMED']
        else:
            row = []
        rows.append(row)
    return dict(targets=catalog, rows=rows)
