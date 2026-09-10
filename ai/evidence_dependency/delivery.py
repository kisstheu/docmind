"""Project validated values into the existing decision/table renderer exactly once."""
from __future__ import annotations

from decimal import Decimal
import re

from ai.decision_result import DecisionResult
from ai.table_presentation import StructuredTable
from .executor import CONFIRMED, CONFLICT, HYPOTHETICAL, display_number, resolve_refs


def _value_with_unit(result):
    text = str(result.value)
    if not result.unit or text.endswith(result.unit):
        return text
    if isinstance(result.value, str) and not re.fullmatch(r"[-\d.,~–—−]+", text):
        return f"{text}（{result.unit}）"
    return text + result.unit


def deliver(execution, candidates, question):
    p, values = execution.proposal, execution.results
    sources = {c.source_id: c for c in candidates}
    names = {o.id: o.label for o in p.objects}
    facts = {f.id: f for f in p.facts}
    derivations = {d.id: d for d in p.derivations}
    choices = {d.id: d for d in p.decisions}
    # Deliver all validated relevant facts; omission from display must not erase useful independent evidence.
    visible = list(dict.fromkeys([*(f.id for f in p.facts if f.kind != "requirement"), *p.delivery,
                                 *(d.id for d in p.derivations), *(d.id for d in p.decisions)]))
    rows, actions, missing, differences, used = [], [], [], [], []
    selected = None
    confirmed_matches = {d.subject for d in p.decisions if d.op == "all_match"
                         and values[d.id].state == CONFIRMED and values[d.id].value is True}
    rankings = []
    for key in visible:
        r = values.get(key)
        if r is None or key in {x.id for x in p.relations}: continue
        if key in facts and facts[key].kind == "requirement": continue
        name = names.get(r.subject, "比较范围")
        label, confirmed, hypothetical, unknown = r.label, "", "", ""
        if r.state == CONFIRMED:
            if key in facts:
                f = facts[key]
                confirmed = f"{_value_with_unit(r)}（{f.scope}）" if f.scope else _value_with_unit(r)
            elif key in derivations:
                d = derivations[key]
                if d.op in {"match", "mismatch"}:
                    condition = facts[d.inputs[1]]
                    confirmed = ("符合该项条件：" if r.value else "明确不符合该项条件：") + _value_with_unit(values[condition.id])
                elif d.op == "compare": confirmed = "比较成立" if r.value else "比较不成立"
                else:
                    a, b = [values[i] for i in d.inputs]
                    symbol = {"add": "+", "subtract": "−", "multiply": "×", "divide": "÷", "ceil_divide": "÷"}[d.op]
                    expression = f"{display_number(a.value)}{a.unit} {symbol} {display_number(b.value)}{b.unit}"
                    confirmed = f"{display_number(r.value)}{r.unit}（{expression}{'，向上取整' if d.op == 'ceil_divide' else ''}）"
            elif key in choices:
                d = choices[key]
                if d.op == "all_match":
                    label = "用户硬条件"
                    confirmed = "全部已列硬条件符合" if r.value else "有明确不符合的硬条件"
                    if not r.value: differences.append(f"{name}：有明确不符合的用户硬条件；见对应条件判断。")
                else:
                    label = "限定范围比较"
                    scope = "、".join(names[x] for x in d.scope)
                    winners = "、".join(names[x] for x in r.value)
                    confirmed = f"在{scope}之间，{winners}的{values[d.inputs[0]].label}{'最低' if d.op == 'minimum' else '最高'}。"
                    rankings.append(confirmed)
                    if len(r.value) == 1 and r.value[0] in confirmed_matches: selected = r.value[0]
        elif r.state == HYPOTHETICAL:
            value = ("该项条件成立" if r.value else "该项条件不成立") if isinstance(r.value, bool) else f"{display_number(r.value)}{r.unit}"
            hypothetical = f"若{r.premise or '所列输入及对应关系得到确认'}，则{value}；当前前提尚未证实，不参与现实排名或选择。"
        else:
            unknown = "存在证据冲突，待确认" if r.state == CONFLICT else "待确认，现有证据不足"
            if key in facts:
                # Preserve source statements without relabelling them as the rejected business fact.
                try:
                    quotes, paths = resolve_refs(facts[key].refs, sources, question)
                    confirmed = "原文陈述（不确认候选归属）：「" + "；".join(quotes) + "」"
                    used.extend(paths)
                except ValueError: pass
            if key in choices: label = "用户条件匹配" if choices[key].op == "all_match" else "限定范围比较"
            missing.append(f"{name}：{label}待确认。")
        if not label: label = "事实依据"
        action = None
        if unknown:
            action = len(actions) + 1
            # No model-proposed action text, priority or unconditional purchase side channel.
            actions.append(f"{action}. 核实{name}的{label}及其主体、适用范围和对应关系；确认前不据此选择。")
        used.extend(r.sources)
        rows.append((name, label, confirmed or "—", hypothetical or "—", unknown or "—",
                     actions[action-1] if action else "—", "\n".join(r.sources) or "待绑定"))
    # Missing match proposals are missing dependencies, not permission to omit user conditions.
    hard = [f for f in p.facts if f.kind == "requirement" and f.requirement == "hard"
            and values[f.id].state == CONFIRMED]
    for obj in p.objects:
        covered = {node.inputs[1] for node in p.derivations
                   if node.subject == obj.id and node.op in {"match", "mismatch"}
                   and values[node.id].state == CONFIRMED}
        gaps = [f for f in hard if f.id not in covered]
        if not gaps:
            continue
        condition = "；".join(f"{f.attribute}：{_value_with_unit(values[f.id])}" for f in gaps)
        gap = f"{obj.label}尚未建立完整匹配依据：{condition}。"
        missing.append(gap)
        step = f"{len(actions)+1}. 核实{obj.label}是否满足这些用户条件：{condition}；确认前不据此选择。"
        actions.append(step)
        paths = tuple(dict.fromkeys(path for value in values.values() if value.subject == obj.id for path in value.sources))
        rows.append((obj.label, "用户硬条件", "—", "—", "待确认："+condition, step, "\n".join(paths) or "待绑定"))
    selected_paths = tuple(dict.fromkeys(pth for r in values.values() if r.subject == selected and r.state == CONFIRMED for pth in r.sources)) if selected else ()
    conclusion = (f"在已列比较范围和已确认硬条件下，可选择{names[selected]}。" if selected
                  else "当前保留已确认信息和条件性结果；尚无足够依据确定选择，先核实表中缺口。")
    if not selected and confirmed_matches and not rankings:
        conclusion = "已确认" + "、".join(names[o.id] for o in p.objects if o.id in confirmed_matches) + "满足已列用户硬条件；具体事实和适用范围见表。"
    elif not selected and not p.decisions and not hard and any(
            values[d.id].state == CONFIRMED for d in p.derivations):
        conclusion = "已完成有明确依据的计算，结果和各自适用范围见表。"
    if rankings: conclusion += "\n" + "\n".join(rankings)
    if not actions:
        actions.append("1. 按表中来源及适用范围核对已确认结果；条件性结果须先确认前提。")
    table = StructuredTable(("对象", "项目", "已确认值或来源陈述", "条件性推演", "待确认", "下一步", "来源"), tuple(rows)) if rows else None
    return DecisionResult(
        conclusion=conclusion, selected_candidate=names.get(selected),
        reason="计算结果最多显示两位小数，按四舍五入显示。" if derivations else "",
        comparison_requested=True, comparison_table=table,
        differences="\n".join(differences), missing_information="\n".join(dict.fromkeys(missing)),
        next_actions="\n".join(actions), source_files=tuple(dict.fromkeys(used)),
        selected_source_files=selected_paths,
    )
