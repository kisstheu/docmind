from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai.decision_result import parse_decision_result, render_decision_result
from app.dialog.state_machine import ConversationState, detect_dialog_event
from app.dialog.task_semantics import needs_multi_source_decision_delivery
from public_tests.state.test_repo_meta_generic_file_list_scope_contract import (
    _RecordingLogger,
    _run_turns,
)


# The same delivery contract uses incompatible conditions and missing facts from
# three unrelated domains. None of these dimensions are Core routing signals.
CASES = [
    (
        "岗位", "我希望远程工作。比较这几个岗位，给我推荐一个。",
        ["远程工作；考核周期未明确", "必须现场工作，不符合远程目标", "混合办公，每周到场两天", "远程工作，夜间值守"],
    ),
    (
        "合同", "我要求一年内结束。比较这几个合同方案，给我推荐一个。",
        ["期限十个月；验收责任未明确", "期限两年，不符合一年内结束目标", "期限六个月，不含维护费用", "期限九个月，费用包含维护"],
    ),
    (
        "设备", "我需要离线运行。比较这几个设备，给我推荐一个。",
        ["支持离线；续航时长未明确", "必须联网，不符合离线目标", "支持离线，额定功率十瓦", "支持离线，峰值功率十五瓦，口径不同"],
    ),
]


def _response(kind, paths, facts, *, selected=True):
    comparison = "\n".join(
        ["| 对象 | 主要事实 | 来源 |", "|---|---|---|"]
        + [f"| 合成{kind}{letter} | {fact} | {path} |" for letter, fact, path in zip("甲乙丙丁", facts, paths)]
    )
    return (
        f"推荐结论：{'甲相对更合适，但仍需确认缺失条件' if selected else '暂无足够匹配的候选'}\n"
        f"推荐对象：{'合成' + kind + '甲' if selected else '无'}\n"
        "推荐理由：依据下面各项比较作出判断，未明确的条件不视为已满足。\n"
        f"横向比较：\n{comparison}\n"
        f"差异与异常：{facts[1]}，来源【{paths[1]}】。{facts[3]}，来源【{paths[3]}】。\n"
        f"待确认信息：{facts[0]}，来源【{paths[0]}】。\n"
        f"推荐对象来源：{paths[0] if selected else '无'}\n"
        # Even an abbreviated footer must not erase other in-section citations.
        f"来源文件：{paths[0]}"
    )


@pytest.mark.parametrize("kind,question,facts", CASES)
def test_first_turn_comparison_survives_one_selection_through_real_runner(
    monkeypatch, tmp_path, kind, question, facts,
):
    from app import chat_loop as runtime

    paths = [f"{index:02d}_合成{kind}{letter}.md" for index, letter in enumerate("甲乙丙丁", 1)]
    calls = []

    def generate_content(*, model, contents, config=None):
        calls.append(contents)
        assert "【多来源比较与决策任务】" in contents
        assert "仅写被推荐对象的来源文件" not in contents
        assert "【当前焦点】" not in contents
        for path, fact in zip(paths, facts):
            assert path in contents
            assert fact in contents
        return SimpleNamespace(text=_response(kind, paths, facts))

    _run_turns(
        monkeypatch, tmp_path, questions=[question], repo_paths=paths,
        repo_chunks=facts, state=ConversationState(),
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
    )
    state = runtime.conversation_state
    assert len(calls) == 1
    assert state.last_route == "normal_retrieval"
    assert state.last_selected_candidate == f"合成{kind}甲"
    assert state.last_selected_source_files == [paths[0]]
    assert state.last_answer_source_files == paths
    assert state.last_result_set_items is None  # comparison is not a selectable enumeration
    for fact, path in zip(facts, paths):
        assert fact in state.last_answer_text
        assert path in state.last_answer_text
    assert "|---|---|---|" in state.last_answer_text
    assert "方向匹配不代表" not in state.last_answer_text
    assert "这些从你目前提供的信息里还无法确认" not in state.last_answer_text
    assert detect_dialog_event("详细分析一下", state, _RecordingLogger()).name == "selected_candidate_followup"


@pytest.mark.parametrize("kind,question,facts", CASES)
def test_no_selection_keeps_comparison_and_only_cited_evidence(kind, question, facts):
    paths = [f"{index:02d}_合成{kind}{letter}.md" for index, letter in enumerate("甲乙丙丁", 1)]
    result = parse_decision_result(
        _response(kind, paths, facts, selected=False), user_question=question,
        comparison_source_files=[*paths, "未参与比较.md"],
    )
    assert result is not None
    assert result.selected_candidate is None
    assert result.selected_source_files == ()
    assert result.source_files == tuple(paths)
    answer = render_decision_result(result)
    for fact in facts:
        assert fact in answer
    assert "未参与比较.md" not in answer
    assert "我更推荐" not in answer


@pytest.mark.parametrize("kind,question,facts", CASES)
def test_comparison_sources_require_exact_context_identity(kind, question, facts):
    paths = [f"{index:02d}_合成{kind}{letter}.md" for index, letter in enumerate("甲乙丙丁", 1)]
    raw = _response(kind, paths, facts).replace(paths[0], "臆造来源.md")
    result = parse_decision_result(raw, comparison_source_files=paths)
    assert result.selected_candidate is None
    assert result.source_files == tuple(paths[1:])
    assert "臆造来源.md" not in result.source_files
    assert all(fact in render_decision_result(result) for fact in facts)


@pytest.mark.parametrize("source_count", [1, 3])
def test_single_object_decision_keeps_existing_delivery(monkeypatch, tmp_path, source_count):
    from app import chat_loop as runtime

    question = "就方案甲给我一个推荐建议。"
    paths = [f"方案甲说明{index}.md" for index in range(source_count)]
    calls = []

    def generate_content(*, model, contents, config=None):
        calls.append(contents)
        assert "【多来源比较与决策任务】" not in contents
        assert "横向比较：" not in contents
        return SimpleNamespace(text=f"推荐对象：方案甲\n推荐理由：约束符合。\n来源文件：{paths[0]}")

    _run_turns(
        monkeypatch, tmp_path, questions=[question], repo_paths=paths,
        repo_chunks=["方案甲支持离线运行。"] * source_count, state=ConversationState(),
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
    )
    assert len(calls) == 1
    assert runtime.conversation_state.last_selected_candidate == "方案甲"
    assert "|" not in runtime.conversation_state.last_answer_text
    assert runtime.conversation_state.last_answer_source_files is None


@pytest.mark.parametrize(
    "question,paths,expected",
    [
        ("比较这些岗位并推荐一个", ["甲.md", "乙.md"], True),
        ("对比这些合同的优劣", ["甲.md", "乙.md"], True),
        ("这几个设备哪个更适合？", ["甲.md", "乙.md"], True),
        ("比较这个对象与我的条件", ["甲.md"], False),
        ("比较这个对象与我的条件", ["甲.md", "甲.md"], False),
        ("推荐一个方案", ["甲.md", "乙.md"], False),
        ("这个设备的参数是什么？", ["甲.md", "乙.md"], False),
        ("这份合同是否推荐使用某种条款？", ["甲.md", "乙.md"], False),
    ],
)
def test_delivery_needs_comparison_intent_and_multiple_distinct_sources(question, paths, expected):
    assert needs_multi_source_decision_delivery(question, paths) is expected


def test_delivery_has_explicit_rollback(monkeypatch):
    monkeypatch.setenv("DOCMIND_MULTI_SOURCE_DECISION_DELIVERY", "0")
    assert not needs_multi_source_decision_delivery("比较这些方案并推荐一个", ["甲.md", "乙.md"])


@pytest.mark.parametrize("kind,question,facts", CASES)
def test_comparison_before_decision_fields_is_not_discarded(kind, question, facts):
    paths = [f"{index:02d}_合成{kind}{letter}.md" for index, letter in enumerate("甲乙丙丁", 1)]
    preamble = "\n".join(f"{fact}，来源【{path}】" for fact, path in zip(facts, paths))
    raw = (
        f"{preamble}\n\n推荐对象：合成{kind}甲\n推荐理由：相对接近目标。\n"
        f"未被证据证明的要求：{facts[0]}\n明显差距或风险：{facts[1]}\n来源文件：{paths[0]}"
    )
    result = parse_decision_result(raw, user_question=question, comparison_source_files=paths)
    assert result is not None
    assert result.source_files == tuple(paths)
    answer = render_decision_result(result)
    assert preamble in answer
    assert facts[0] in answer and facts[1] in answer
    assert "方向匹配不代表" not in answer
    # An older format without separate selection provenance cannot narrow followups safely.
    assert result.selected_source_files == ()


@pytest.mark.parametrize("kind,question,facts", CASES)
def test_unstructured_comparison_remains_visible_without_inventing_selection(kind, question, facts):
    paths = [f"合成{kind}{letter}.md" for letter in "甲乙丙丁"]
    raw = "\n".join(f"{fact}，来源【{path}】" for fact, path in zip(facts, paths))
    result = parse_decision_result(raw, user_question=question, comparison_source_files=paths)
    assert result is not None
    assert raw in render_decision_result(result)
    assert result.source_files == tuple(paths)
    assert result.selected_candidate is None


@pytest.mark.parametrize("kind,question,facts", CASES)
def test_selected_detail_retrieves_only_candidate_sources_after_comparison(
    monkeypatch, tmp_path, kind, question, facts,
):
    paths = [f"{index:02d}_合成{kind}{letter}.md" for index, letter in enumerate("甲乙丙丁", 1)]
    calls = []

    def generate_content(*, model, contents, config=None):
        calls.append(contents)
        if len(calls) == 1:
            return SimpleNamespace(text=_response(kind, paths, facts))
        assert "【已选对象追问】" in contents
        context = contents.split("【参考片段】:", 1)[1].split("【用户最新提问】", 1)[0]
        assert paths[0] in context
        assert all(path not in context for path in paths[1:])
        return SimpleNamespace(text=f"已选对象仍需确认：{facts[0]}。来源：{paths[0]}")

    _run_turns(
        monkeypatch, tmp_path, questions=[question, "详细分析一下"], repo_paths=paths,
        repo_chunks=facts, state=ConversationState(),
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
    )
    assert len(calls) == 2
