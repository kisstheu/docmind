"""Dependency closure and bounded Decimal execution; never execute generated code."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_HALF_UP, localcontext
import re

from .protocol import Proposal, Ref

CONFIRMED, UNKNOWN, CONFLICT, HYPOTHETICAL = "CONFIRMED", "UNKNOWN", "CONFLICT", "HYPOTHETICAL"


@dataclass
class Result:
    id: str
    state: str = UNKNOWN
    value: object = None
    unit: str = ""
    subject: str = ""
    label: str = ""
    sources: tuple[str, ...] = ()
    inputs: tuple[str, ...] = ()
    reason: str = "依赖待确认"
    premise: str = ""
    user_confirmations: tuple[str, ...] = ()


@dataclass
class Execution:
    proposal: Proposal
    results: dict[str, Result]
    checks: list[dict]
    verdicts: dict
    diagnostics: dict = field(default_factory=dict)


def resolve_refs(refs, sources, question):
    if not refs:
        raise ValueError("missing references")
    quotes, paths = [], []
    for ref in refs:
        source = sources.get(ref.source_id)
        if source is None and ref.source_id != "user":
            raise ValueError("unknown source")
        lines = (question if ref.source_id == "user" else source.text).splitlines()
        a, b = ref.line_start, ref.line_end
        if type(a) is not int or type(b) is not int or not 1 <= a <= b <= len(lines):
            raise ValueError("invalid line range")
        quote = _resolve_source_quote(lines, a, b, ref.quote)
        if quote is None:
            raise ValueError("quote mismatch")
        quotes.append(quote)
        if source:
            paths.append(source.path)
    return quotes, tuple(dict.fromkeys(paths))


def _compact_layout_text(value):
    """Remove presentation whitespace without changing any semantic character."""
    return re.sub(r"\s+", "", value or "")


def _resolve_source_quote(lines, line_start, line_end, expected):
    """Bind an exact quote despite source wrapping and a one-line boundary drift."""
    needle = _compact_layout_text(expected)
    if not needle:
        return None
    selected = "\n".join(lines[line_start - 1:line_end])
    if needle in _compact_layout_text(selected):
        return selected

    # OCR/PDF wrapping commonly makes a model include one adjacent physical line.
    # Accept that bounded drift only when the matched quote still overlaps the
    # declared range; a quote found solely on a neighboring line remains invalid.
    lower = max(0, line_start - 2)
    upper = min(len(lines), line_end + 1)
    compact_lines = [_compact_layout_text(line) for line in lines[lower:upper]]
    expanded = "".join(compact_lines)
    claimed_start = sum(len(line) for line in compact_lines[:line_start - 1 - lower])
    claimed_end = claimed_start + sum(
        len(line) for line in compact_lines[line_start - 1 - lower:line_end - lower]
    )
    offset = expanded.find(needle)
    while offset >= 0:
        if offset < claimed_end and offset + len(needle) > claimed_start:
            return "\n".join(lines[lower:upper])
        offset = expanded.find(needle, offset + 1)
    return None


def number(value):
    if not isinstance(value, (str, Decimal)) or not re.fullmatch(r"-?\d{1,18}(?:\.\d{1,12})?", str(value)):
        raise ValueError("invalid finite decimal")
    result = Decimal(value)
    if not result.is_finite() or abs(result) > Decimal("1e18"):
        raise ValueError("out of range")
    return result


def fact_value(fact):
    if not fact.unit:
        return fact.value
    try:
        return number(fact.value)
    except ValueError:
        try:
            # Exact declared suffix only: this neither converts a unit nor invents a sales basis.
            return number(fact.value[:-len(fact.unit)]) if fact.value.endswith(fact.unit) else fact.value
        except ValueError:
            return fact.value


def unit_parts(unit):
    # Formal cancellation of * and /. Never synonym conversion or numeric scale guessing.
    if not isinstance(unit, str) or not unit or len(unit) > 30:
        raise ValueError("missing unit")
    tokens = re.split(r"([*/])", unit)
    parts, sign = {}, 1
    for token in tokens:
        if token in {"*", "/"}:
            sign = 1 if token == "*" else -1
        elif token == "1":
            continue
        elif not token or re.search(r"\s|[()=]", token):
            raise ValueError("unsupported unit")
        else:
            parts[token] = parts.get(token, 0) + sign
    return {k: v for k, v in parts.items() if v}


def calculate(op, left, right, output_unit, comparator="eq"):
    a, b = number(left.value), number(right.value)
    ua, ub = unit_parts(left.unit), unit_parts(right.unit)
    if op in {"add", "subtract", "compare"}:
        if ua != ub:
            raise ValueError("incompatible units")
        expected = ua
    else:
        expected = dict(ua)
        for k, v in ub.items():
            expected[k] = expected.get(k, 0) + (v if op == "multiply" else -v)
        expected = {k: v for k, v in expected.items() if v}
    if op != "compare" and unit_parts(output_unit) != expected:
        raise ValueError("unsupported unit conversion")
    with localcontext() as ctx:
        ctx.prec = 40
        if op == "add": value = a + b
        elif op == "subtract": value = a - b
        elif op == "multiply": value = a * b
        elif op in {"divide", "ceil_divide"}:
            if not b or (op == "ceil_divide" and (a < 0 or b <= 0)):
                raise ValueError("invalid divisor or coverage target")
            value = a / b
            if op == "ceil_divide": value = value.to_integral_value(rounding=ROUND_CEILING)
        elif op == "compare":
            return {"eq": a == b, "lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b}[comparator]
        else: raise ValueError("unsupported operation")
        if not value.is_finite() or abs(value) > Decimal("1e18"):
            raise ValueError("out of range")
        # Keep precision for downstream arithmetic; round only in delivery.
        return value


def display_number(value):
    if not isinstance(value, Decimal):
        return str(value)
    with localcontext() as ctx:
        ctx.prec = 40
        return format(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f").rstrip("0").rstrip(".")


def execute(proposal, candidates, question, verifier, *, user_evidence=None):
    from .user_evidence import UserEvidence
    if user_evidence is not None and not isinstance(user_evidence, UserEvidence):
        raise ValueError("user evidence must come from the validated caller input")
    user_facts = user_evidence.facts if user_evidence else {}
    user_ops = user_evidence.operations if user_evidence else {}
    sources = {c.source_id: c for c in candidates}
    if len(sources) != len(candidates):
        raise ValueError("duplicate source ID")
    objects = {o.id: o.label for o in proposal.objects}
    nodes = [*proposal.facts, *proposal.relations, *proposal.derivations, *proposal.decisions]
    index = {n.id: n for n in nodes}
    if (len(index) != len(nodes) or len(objects) != len(proposal.objects)
            or set(objects) & set(index) or "USER" in objects or "__requirements__" in index):
        raise ValueError("duplicate or reserved ID")
    results = {n.id: Result(n.id, subject=getattr(n, "subject", ""),
                            label=getattr(n, "attribute", getattr(n, "label", ""))) for n in nodes}
    checks = []
    valid_facts = set()
    facts = {f.id: f for f in proposal.facts}
    relations = {r.id: r for r in proposal.relations}
    derivations = {d.id: d for d in proposal.derivations}
    decisions = {d.id: d for d in proposal.decisions}
    descriptions = {}
    for f in proposal.facts:
        result = results[f.id]
        result.unit = f.unit
        try:
            supplied = user_facts.get(f.id)
            quotes, paths = ([], supplied['sources']) if supplied else resolve_refs(f.refs, sources, question)
            if f.kind == "requirement":
                if f.subject != "USER" or any(r.source_id != "user" for r in f.refs):
                    raise ValueError("requirement is not from user")
            elif f.subject not in objects or any(r.source_id == "user" for r in f.refs) and f.kind == "source":
                raise ValueError("source subject/reference invalid")
            if f.kind != "requirement" and f.requirement != "none":
                raise ValueError("invented requirement")
            result.sources = paths
            # Presence is necessary for direct values, never sufficient for their relation.
            if (not supplied and f.kind != "hypothesis"
                    and _compact_layout_text(f.value) not in _compact_layout_text("\n".join(quotes))):
                raise ValueError("value not stated in source")
            descriptions[f.id] = dict(subject=objects.get(f.subject, "USER"), attribute=f.attribute,
                                      value=supplied['value'] if supplied else str(fact_value(f)), unit=f.unit, scope=f.scope, kind=f.kind,
                                      requirement=f.requirement)
            checks.append(dict(id=f.id, kind="fact", claim=descriptions[f.id], refs=[r.model_dump() for r in f.refs]))
            valid_facts.add(f.id)
        except ValueError as exc:
            result.reason = str(exc)

    # Describe transitive inputs without trusting model's optional requires list.
    def describe(node_id, visiting=()):
        if node_id in visiting or len(visiting) > 32:
            raise ValueError("cyclic dependencies")
        if node_id in descriptions:
            return descriptions[node_id]
        d = derivations.get(node_id)
        if d is None:
            raise ValueError("dangling or invalid input")
        return dict(subject=objects.get(d.subject), label=d.label, op=d.op, unit=d.unit,
                    inputs=[describe(i, (*visiting, node_id)) for i in d.inputs])

    # Fetch all transitive facts as well as relation refs, including adjacent selected chunks.
    def inherited_refs(i, seen=()):
        if i in seen: raise ValueError("cycle")
        if i in facts: return facts[i].refs
        if i in derivations:
            prev = derivations[i]
            return [x for j in prev.inputs for x in inherited_refs(j, (*seen, i))]
        raise ValueError("missing input")

    checked_ops = set()
    for d in proposal.derivations:
        out = results[d.id]
        out.inputs = tuple(d.inputs) + (d.relation,)
        out.unit = d.unit
        try:
            r = relations.get(d.relation)
            if r is None or set(r.inputs) != set(d.inputs) or len(set(d.inputs)) != 2:
                raise ValueError("required relation/input missing")
            if d.subject not in objects:
                raise ValueError("invalid result subject")
            quotes, paths = ([], ()) if d.id in user_ops else resolve_refs(r.refs, sources, question)
            if d.op in {"match", "mismatch"}:
                target = facts.get(d.inputs[1])
                if not target or target.kind != "requirement" or d.unit:
                    raise ValueError("match requires user condition")
            else:
                if not d.unit and d.op != "compare":
                    raise ValueError("missing result unit")
            refs = list(r.refs) + [ref for i in d.inputs for ref in inherited_refs(i)]
            checks.append(dict(id=d.id, kind="hypothetical_premise" if r.hypothetical or d.hypothetical else "condition_match" if d.op in {"match", "mismatch"} else "operation_premise", claim={
                **describe(d.id), "relation": ("如果确认以下对应关系成立：" + r.claim) if r.hypothetical or d.hypothetical else r.claim, "hypothetical": d.hypothetical,
                "relation_hypothetical": r.hypothetical, "comparator": d.comparator,
            }, refs=[ref.model_dump() for ref in refs]))
            checked_ops.add(d.id)
            out.sources = tuple(dict.fromkeys((*paths, *(p for i in d.inputs for p in results[i].sources))))
            out.premise = r.claim if r.hypothetical or d.hypothetical else ""
        except (ValueError, KeyError) as exc:
            out.reason = str(exc)
    requirements = [f for f in proposal.facts if f.kind == "requirement"]
    if decisions:
        checks.append(dict(id="__requirements__", kind="requirement_coverage",
                           claim=[descriptions.get(f.id, {}) for f in requirements],
                           refs=[Ref(source_id="user", line_start=1, line_end=max(1, len(question.splitlines())), quote=question).model_dump()]))
    # Enumerate comparison premises in the same bounded local batch schedule.
    rank_checks = []
    for d in proposal.decisions:
        if d.op == "all_match": continue
        try:
            rank_checks.append(dict(id=d.id, kind="comparison_scope", claim={
                "op": d.op, "scope": [objects[i] for i in d.scope],
                "inputs": [describe(i) for i in d.inputs]},
                refs=[ref.model_dump() for i in d.inputs for ref in inherited_refs(i)]))
        except (ValueError, KeyError): pass
    checks.extend(rank_checks)
    verdicts = verifier(checks, tuple(candidates), question) if checks else {}
    if not isinstance(verdicts, dict): verdicts = {}
    for key, supplied in user_facts.items():
        if key in valid_facts:
            verdicts[key] = dict(status="SUPPORTED", basis="用户核实，在本任务采用")
    for key in user_ops:
        if key in checked_ops:
            verdicts[key] = dict(status="SUPPORTED", basis="用户核实对应关系，在本任务采用",
                                 comparison_operator="semantic")

    def state_for(key):
        verdict = verdicts.get(key, {})
        return {"SUPPORTED": CONFIRMED, "CONFLICT": CONFLICT}.get(verdict.get("status"), UNKNOWN)

    for f in proposal.facts:
        if f.id not in valid_facts: continue
        result = results[f.id]
        result.state = state_for(f.id)
        result.reason = verdicts.get(f.id, {}).get("basis", "语义核验未完成")
        if result.state == CONFIRMED:
            supplied = user_facts.get(f.id)
            result.value = fact_value(f.model_copy(update={'value': supplied['value']})) if supplied else fact_value(f)
            if supplied:
                result.user_confirmations = (supplied['record']['confirmation_id'],)
            if f.kind == "hypothesis":
                result.state, result.premise = HYPOTHETICAL, f.scope

    # A correction adopts this fact's new value, but does not erase independent
    # contradictory facts for the exact same subject, attribute and scope.
    groups = {}
    for f in proposal.facts:
        r = results[f.id]
        if f.kind == 'source' and r.state == CONFIRMED:
            groups.setdefault((f.subject, f.attribute, f.scope, f.unit), []).append(r)
    for group in groups.values():
        if any(r.user_confirmations for r in group) and len({str(r.value) for r in group}) > 1:
            for r in group:
                r.state, r.reason = CONFLICT, '用户核实与同一对象、事项、范围的其他证据存在冲突'

    visiting, completed = set(), set()
    def evaluate(key):
        if key not in results: return Result(key, reason="dangling dependency")
        result = results[key]
        if key in completed or key in facts: return result
        if key in visiting: return Result(key, reason="cyclic dependency")
        visiting.add(key)
        if key in derivations and key in checked_ops:
            d = derivations[key]
            r = relations[d.relation]
            args = [evaluate(i) for i in d.inputs]
            states = [a.state for a in args] + [state_for(key)]
            hypothetical = ((r.hypothetical or d.hypothetical) and key not in user_ops) or HYPOTHETICAL in states
            result.premise = "；".join(dict.fromkeys(x for x in [result.premise, *(a.premise for a in args)] if x))
            result.sources = tuple(dict.fromkeys((*result.sources, *(p for a in args for p in a.sources))))
            result.user_confirmations = tuple(dict.fromkeys([
                *(cid for a in args for cid in a.user_confirmations),
                *([user_ops[key]['confirmation_id']] if key in user_ops else [])]))
            if CONFLICT in states: result.state = CONFLICT
            elif UNKNOWN not in states:
                try:
                    # An assumption cannot upgrade into a confirmed operation, even when omitted by the model.
                    if d.op in {"match", "mismatch"}:
                        if args[0].subject != d.subject:
                            raise ValueError("wrong match subject")
                        comparator = verdicts[key].get("comparison_operator", "semantic")
                        if key in user_ops:
                            comparator = user_ops[key].get('comparison_operator', 'semantic')
                        if comparator == "semantic" and all(isinstance(a.value, Decimal) for a in args):
                            raise ValueError("numeric condition requires a checked comparator")
                        value = ((args[0].value == args[1].value if key in user_ops else d.op == "match") if comparator == "semantic" else
                                 calculate("compare", *args, "", comparator))
                    else:
                        value = calculate(d.op, *args, d.unit, d.comparator)
                    result.state = HYPOTHETICAL if hypothetical else CONFIRMED
                    result.value, result.reason = value, verdicts[key].get("basis", "narrow relation check")
                    results[d.relation].state = result.state
                    results[d.relation].reason = result.reason
                except (ValueError, InvalidOperation, KeyError) as exc: result.reason = str(exc)
        elif key in decisions:
            d = decisions[key]
            result.inputs = tuple(d.inputs)
            args = [evaluate(i) for i in d.inputs]
            hard = {f.id for f in requirements if f.requirement == "hard"}
            coverage = state_for("__requirements__") == CONFIRMED
            if d.op == "all_match":
                matches = {derivations[i].inputs[1]: evaluate(i) for i in d.inputs
                           if i in derivations and derivations[i].op in {"match", "mismatch"}
                           and derivations[i].subject == d.subject}
                judgments = {}
                for i in d.inputs:
                    node = derivations.get(i)
                    if node and node.op in {"match", "mismatch"} and node.subject == d.subject:
                        value = evaluate(i)
                        if value.state == CONFIRMED:
                            judgments.setdefault(node.inputs[1], set()).add(value.value)
                if any(len(v) > 1 for v in judgments.values()):
                    result.state, result.reason = CONFLICT, "同一条件存在相互矛盾的判断"
                elif hard and coverage and all(results[i].state == CONFIRMED for i in hard):
                    if any(matches[i].state == CONFIRMED and matches[i].value is False for i in hard & matches.keys()):
                        result.state, result.value = CONFIRMED, False
                    elif hard <= matches.keys() and all(matches[i].state == CONFIRMED and matches[i].value is True for i in hard):
                        result.state, result.value = CONFIRMED, True
            elif (len(args) >= 2 and len(set(d.scope)) == len(d.scope)
                  and set(d.scope) == {a.subject for a in args} and len(args) == len(d.scope)
                  and all(a.state == CONFIRMED and isinstance(a.value, Decimal) for a in args)
                  and len({a.unit for a in args}) == 1 and len({a.label for a in args}) == 1):
                # Equal display labels/units still need a semantic common comparison scope.
                # This check is enumerated separately below; no minimum without its support.
                if state_for(key) == CONFIRMED:
                    best = (min if d.op == "minimum" else max)(a.value for a in args)
                    result.state = CONFIRMED
                    result.value = tuple(a.subject for a in args if a.value == best)
            result.sources = tuple(dict.fromkeys(p for a in args for p in a.sources))
            result.user_confirmations = tuple(dict.fromkeys(cid for a in args for cid in a.user_confirmations))
        visiting.remove(key)
        completed.add(key)
        return result

    for key in (*derivations, *decisions): evaluate(key)
    return Execution(proposal, results, checks, verdicts, {
        "check_count": len(checks), "checked_operations": sorted(checked_ops),
        "states": {key: value.state for key, value in results.items()},
    })
