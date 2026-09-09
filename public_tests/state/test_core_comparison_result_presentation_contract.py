from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ai.decision_result import build_comparison_generation_config, parse_decision_result, render_decision_result
from ai.prompt_builder import build_final_prompt
from app.dialog.state_machine import ConversationState, detect_dialog_event
from app.dialog.task_semantics import needs_multi_source_decision_delivery
from public_tests.state.test_repo_meta_generic_file_list_scope_contract import (
    _RecordingLogger,
    _run_turns,
)


# Scores and incompatible requirements are synthetic evidence, never Core rules.
CASES = [
    ("岗位", "远程工作", "办公方式待确认", "必须现场工作"),
    ("合同方案", "一年内结束", "履行期限待确认", "履行期限两年"),
    ("设备", "离线运行", "离线能力待确认", "必须联网"),
]


def _fixture(kind, condition, missing, incompatible, *, table=True):
    paths = [f"{index:02d}_合成{kind}{letter}.md" for index, letter in enumerate("甲乙丙", 1)]
    facts = [f"评估分数95；{missing}", f"评估分数90；已明确{condition}", f"评估分数98；{incompatible}"]
    conclusion = (
        f"先确认合成{kind}甲是否{condition}：甲95分，高于乙90分，但{missing}；"
        f"若确认满足则优先甲，否则乙是在已明确{condition}的对象中更稳妥的选择。"
    )
    rows = [
        f"合成{kind}甲 | 潜在更优，待确认 | {facts[0]} | 确认{condition} | {paths[0]}",
        f"合成{kind}乙 | 条件明确，可作备选 | {facts[1]} | 甲不满足时选择乙 | {paths[1]}",
        f"合成{kind}丙 | 当前不适用 | {facts[2]} | 本轮排除 | {paths[2]}",
    ]
    comparison = (
        "| 对象 | 当前状态 | 关键事实 | 下一步 | 来源 |\n|---|---|---|---|---|\n"
        + "\n".join(f"| {row} |" for row in rows)
        if table else "\n".join(row.replace(" | ", "；") for row in rows)
    )
    response = (
        f"推荐结论：{conclusion}\n推荐对象：合成{kind}乙\n"
        f"推荐理由：乙的条件已明确，但不能把乙写成整体最高，也不能把甲写成最终最优。\n"
        f"横向比较：\n{comparison}\n"
        f"差异与异常：丙{incompatible}，依据【{paths[2]}】。\n"
        f"待确认信息：甲{missing}，依据【{paths[0]}】。\n"
        f"下一步行动：先确认甲；若不满足{condition}则选择乙。\n"
        f"推荐对象来源：{paths[1]}\n来源文件：" + "、".join(paths)
    )
    return paths, facts, conclusion, response


@pytest.mark.parametrize("kind,condition,missing,incompatible", CASES)
@pytest.mark.parametrize("request_kind", ["decision", "action"])
def test_comparison_presentation_and_selected_detail_through_runner(
    monkeypatch, tmp_path, capsys, kind, condition, missing, incompatible, request_kind,
):
    from app import chat_loop as runtime

    paths, facts, conclusion, response = _fixture(kind, condition, missing, incompatible)
    question = (
        f"我要求{condition}。比较这几个{kind}并推荐一个。"
        if request_kind == "decision" else
        f"我要求{condition}。这里是一批{kind}资料，帮我找出可用对象，告诉我优先考虑谁，下一步分别确认什么。"
    )
    state = ConversationState()
    event = detect_dialog_event(question, state, _RecordingLogger())
    assert event.name == f"{request_kind}_request"  # presentation does not reroute actions
    calls = []
    first_states = []

    def generate_content(*, model, contents, config=None):
        calls.append(contents)
        if len(calls) == 1:
            assert "【多来源比较与决策任务】" in contents
            assert "先在回答前部给出简短建议和关键取舍" in contents
            assert "默认使用紧凑 Markdown 表格" in contents
            assert "不强制制造表格" in contents
            assert "同一句保留比较范围、指标和必要条件" in contents
            assert "未选对象在关键指标上更优但缺条件" in contents
            assert "备选对象的下一步不得写成无条件立即执行" in contents
            assert "补充核实事项不能升级为用户未要求的硬性门槛" in contents
            assert "每条独立换行" in contents
            assert "next_actions（下一步行动）" in contents
            assert "尽量不用列表" not in contents
            assert "事实或异常、缺失判断旁可见" in contents
            assert all(path in contents for path in paths)
            parsed = parse_decision_result(response, comparison_source_files=paths)
            payload = {
                key: getattr(parsed, key) for key in (
                    "conclusion", "selected_candidate", "reason", "comparison", "differences",
                    "missing_information", "next_actions", "selected_source_files", "source_files",
                )
            }
            assert config.response_mime_type == "application/json"
            return SimpleNamespace(text=json.dumps(payload, ensure_ascii=False))
        first_states.append(runtime.conversation_state.last_answer_text)
        assert runtime.conversation_state.last_answer_source_files == paths
        assert runtime.conversation_state.last_selected_source_files == [paths[1]]
        assert runtime.conversation_state.last_result_set_items is None
        assert "【已选对象追问】" in contents
        context = contents.split("【参考片段】:", 1)[1].split("【用户最新提问】", 1)[0]
        assert paths[1] in context and all(path not in context for path in (paths[0], paths[2]))
        return SimpleNamespace(text=f"乙已明确{condition}，依据【{paths[1]}】。")

    _run_turns(
        monkeypatch, tmp_path, questions=[question, "详细分析一下"], repo_paths=paths,
        repo_chunks=facts, state=state,
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
    )
    assert len(calls) == 2
    answer = first_states[0]
    assert answer.startswith(f"建议：\n\n{conclusion.rstrip(chr(0x3002))}")
    assert "我更推荐" not in answer  # no unconditional recommendation prepended by renderer
    assert all(fact in answer for fact in facts)
    assert all(path in answer for path in paths)
    assert "下一步行动：\n\n先确认甲" in answer
    assert "| 对象 | 当前状态 | 关键事实 | 下一步 | 来源 |" in answer
    assert conclusion.rstrip("。") in capsys.readouterr().out  # actual printed answer, not just bound state


@pytest.mark.parametrize("kind,condition,missing,incompatible", CASES)
def test_compact_non_table_comparison_keeps_conditions_actions_and_source_identity(
    kind, condition, missing, incompatible,
):
    paths, facts, conclusion, response = _fixture(kind, condition, missing, incompatible, table=False)
    result = parse_decision_result(response, comparison_source_files=paths)
    answer = render_decision_result(result)
    assert answer.startswith(f"建议：\n\n{conclusion.rstrip(chr(0x3002))}")
    assert "|" not in answer
    assert all(fact in answer and path in answer for fact, path in zip(facts, paths))
    assert result.source_files == tuple(paths)
    assert result.selected_source_files == (paths[1],)


@pytest.mark.parametrize("kind,condition,missing,incompatible", CASES)
def test_footer_only_sources_are_made_visible_without_narrowing_to_selection(
    kind, condition, missing, incompatible,
):
    paths, _, _, response = _fixture(kind, condition, missing, incompatible)
    # Legacy raw generations may put citations only in the consumed source fields.
    body = response.split("推荐对象来源：", 1)[0]
    for path in paths:
        body = body.replace(path, "")
    result = parse_decision_result(
        body + f"推荐对象来源：{paths[1]}\n来源文件：" + "、".join(paths),
        comparison_source_files=[*paths, "99_未引用资料.md"],
    )
    answer = render_decision_result(result)
    assert all(path in answer for path in paths)
    assert "99_未引用资料.md" not in answer
    assert result.source_files == tuple(paths)
    assert result.selected_source_files == (paths[1],)


@pytest.mark.parametrize("question,expected", [
    ("这批资料帮我看看，优先联系谁？", True),
    ("这些方案帮我分析，哪个优先采用？", True),
    ("这几个对象帮我分析，先确认哪项？", True),
    ("帮我看看这个对象，优先联系谁？", False),
    ("这些文件分别写了哪些优先事项？", False),
    ("这些设备的参数是什么？", False),
    ("这些合同中的优先权是什么意思？", False),
    ("这批岗位资料的联系人是谁？", False),
])
def test_action_presentation_gate_needs_relative_priority_and_collection(question, expected):
    assert needs_multi_source_decision_delivery(question, ["01_甲.md", "02_乙.md"]) is expected
    assert not needs_multi_source_decision_delivery(question, ["01_甲.md"])


def test_action_presentation_rollback(monkeypatch):
    monkeypatch.setenv("DOCMIND_MULTI_SOURCE_DECISION_DELIVERY", "0")
    assert not needs_multi_source_decision_delivery("这批资料帮我找出优先联系谁", ["甲.md", "乙.md"])


@pytest.mark.parametrize("event,question", [
    ("decision_request", "就这个对象给个推荐建议"),
    ("action_request", "帮我查下这份资料的参数"),
    ("result_set_followup", "刚才这些再详细说说"),
])
def test_adjacent_prompts_keep_existing_presentation(event, question):
    prompt = build_final_prompt([], None, "", "合成事实", question, event_name=event)
    assert "【多来源比较与决策任务】" not in prompt
    assert "默认使用紧凑 Markdown 表格" not in prompt
    assert "尽量不用列表" in prompt


@pytest.mark.parametrize("bad_response", [None, " " * 100_000])
def test_action_comparison_output_validation_keeps_previous_state(
    monkeypatch, tmp_path, bad_response,
):
    from app import chat_loop as runtime

    state = ConversationState(last_answer_text="上一轮有效回答")
    _run_turns(
        monkeypatch, tmp_path,
        questions=["这批资料帮我找出可用对象，告诉我优先考虑谁"],
        repo_paths=["01_甲.md", "02_乙.md"], repo_chunks=["合成对象甲", "合成对象乙"],
        state=state,
        client=SimpleNamespace(models=SimpleNamespace(
            generate_content=lambda **_: SimpleNamespace(text=bad_response),
        )),
    )
    assert runtime.conversation_state.last_answer_text == "上一轮有效回答"
    assert runtime.conversation_state.last_answer_source_files is None


def test_action_only_source_is_bound_and_visible():
    raw = "推荐结论：暂不选择。\n下一步行动：核对依据【03_合成资料.md】中的适用范围。"
    result = parse_decision_result(raw, comparison_source_files=["01_甲.md", "03_合成资料.md"])
    assert result.source_files == ("03_合成资料.md",)
    assert "来源：03_合成资料.md" in render_decision_result(result)


def test_comparison_schema_uses_existing_fields_and_preserves_generation_options():
    from google.genai import types

    original = types.GenerateContentConfig(temperature=0.4, max_output_tokens=4096)
    config = build_comparison_generation_config(original, ["01_合成甲.md", "02_合成乙.md"])
    assert original.response_mime_type is None
    assert config.temperature == 0.4
    assert config.max_output_tokens == 4096
    assert config.response_mime_type == "application/json"
    schema = config.response_schema
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["properties"]["source_files"]["items"]["enum"] == ["01_合成甲.md", "02_合成乙.md"]
    assert types.Schema.model_validate(schema)
    assert build_comparison_generation_config({}, ["甲.md", "乙.md"])["max_output_tokens"] == 8192


@pytest.mark.parametrize("invalid", ['{"conclusion": "未完成"}', '{"conclusion":', '{"comparison": []}'])
def test_invalid_structured_comparison_is_not_presented_as_success(monkeypatch, tmp_path, capsys, invalid):
    from app import chat_loop as runtime

    _run_turns(
        monkeypatch, tmp_path, questions=["帮我比较这几个设备并推荐一个"],
        repo_paths=["01_甲.md", "02_乙.md"], repo_chunks=["合成甲支持离线", "合成乙需要联网"],
        state=ConversationState(last_answer_text="上一轮有效回答"),
        client=SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: SimpleNamespace(text=invalid))),
    )
    assert runtime.conversation_state.last_answer_text == "上一轮有效回答"
    assert "本轮未生成可可靠呈现的比较结果" in capsys.readouterr().out


@pytest.mark.parametrize("kind,condition,missing,incompatible", CASES)
def test_structured_no_selection_keeps_branches_and_both_source_scopes_separate(
    kind, condition, missing, incompatible,
):
    paths, _, conclusion, response = _fixture(kind, condition, missing, incompatible)
    parsed = parse_decision_result(response, comparison_source_files=paths)
    payload = {
        "conclusion": conclusion, "selected_candidate": "", "reason": "",
        "comparison": parsed.comparison, "differences": parsed.differences,
        "missing_information": parsed.missing_information, "next_actions": parsed.next_actions,
        "selected_source_files": [], "source_files": paths,
    }
    # JSON key order cannot push the conclusion behind a table or source footer.
    result = parse_decision_result(json.dumps(dict(reversed(list(payload.items())))), comparison_source_files=paths)
    assert result.selected_candidate is None and result.selected_source_files == ()
    assert result.source_files == tuple(paths)
    assert render_decision_result(result).startswith(f"建议：\n\n{conclusion}")
