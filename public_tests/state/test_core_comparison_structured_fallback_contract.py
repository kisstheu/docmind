from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ai.decision_result import (
    build_comparison_prose_fallback, parse_decision_result, render_decision_result,
)
from app.dialog.state_machine import ConversationState, detect_dialog_event
from public_tests.state.test_core_comparison_result_presentation_contract import CASES, _fixture
from public_tests.state.test_repo_meta_generic_file_list_scope_contract import (
    _RecordingLogger, _run_turns,
)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_nonstandard_json_table_value_cannot_bypass_corruption_guard(constant):
    raw = '{"comparison": "甲支持离线，依据【01_甲.md】", "comparison_table": ' + constant + '}'
    assert parse_decision_result(raw, comparison_source_files=["01_甲.md"]) is None
    assert build_comparison_prose_fallback(raw, ["01_甲.md"]) is None


def _drifted_response(kind, condition, missing, incompatible, variant):
    paths, facts, conclusion, raw = _fixture(kind, condition, missing, incompatible)
    if variant == "prose":
        raw = conclusion + "\n" + "\n".join(
            f"{fact}，依据【{path}】。" for fact, path in zip(facts, paths)
        )
    elif variant == "prose_missing_footer":
        raw = raw.split("推荐对象来源：", 1)[0]
    elif variant == "changed_labels":
        raw = raw.split("推荐对象来源：", 1)[0].replace("推荐结论：", "建议与取舍：").replace(
            "推荐对象：", "可考虑对象：",
        )
    else:
        result = parse_decision_result(raw, comparison_source_files=paths)
        payload = {
            key: getattr(result, key) for key in (
                "conclusion", "selected_candidate", "reason", "comparison", "differences",
                "missing_information", "next_actions", "selected_source_files", "source_files",
            )
        }
        for key in {
            "missing_candidate": ("selected_candidate",),
            "missing_selection_sources": ("selected_source_files",),
            "missing_source_footer": ("selected_source_files", "source_files"),
            "missing_actions": ("next_actions",),
            "fenced_partial": ("selected_candidate",),
        }[variant]:
            del payload[key]
        raw = json.dumps(dict(reversed(list(payload.items()))), ensure_ascii=False)
        if variant == "fenced_partial":
            raw = f"```json\n{raw}\n```"
    return paths, facts, conclusion, raw


@pytest.mark.parametrize("kind,condition,missing,incompatible", CASES)
@pytest.mark.parametrize("request_kind", ["decision", "action"])
@pytest.mark.parametrize("variant", [
    "prose", "prose_missing_footer", "changed_labels", "missing_candidate",
    "missing_selection_sources", "missing_source_footer", "missing_actions", "fenced_partial",
])
def test_format_drift_delivers_comparison_and_clears_stale_selection(
    monkeypatch, tmp_path, capsys, kind, condition, missing, incompatible, request_kind, variant,
):
    from app import chat_loop as runtime

    paths, facts, conclusion, raw = _drifted_response(kind, condition, missing, incompatible, variant)
    question = (
        f"我要求{condition}。比较这几个{kind}并推荐一个。" if request_kind == "decision" else
        f"我要求{condition}。这批{kind}帮我找出可用对象，告诉我优先考虑谁，下一步分别确认什么。"
    )
    state = ConversationState(
        last_answer_text="上一轮有效回答", last_selected_candidate="旧对象",
        last_selected_source_files=[paths[0]],
    )
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        assert "【多来源比较与决策任务】" in kwargs["contents"]
        return SimpleNamespace(text=raw)

    _run_turns(
        monkeypatch, tmp_path, questions=[question], repo_paths=[*paths, "99_无关资料.md"],
        repo_chunks=[*facts, "与比较无关的合成记录"], state=state,
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
    )
    updated = runtime.conversation_state
    output = capsys.readouterr().out
    assert len(calls) == 1  # no second generation chain for format drift
    assert "请重试" not in output
    assert conclusion.rstrip("。") in output
    assert all(fact in output and path in output for fact, path in zip(facts, paths))
    assert "来源：" + "、".join(paths) in updated.last_answer_text
    assert updated.last_answer_source_files == paths
    assert "99_无关资料.md" not in updated.last_answer_text
    assert updated.last_selected_candidate is None
    assert updated.last_selected_source_files is None
    assert updated.last_result_set_items is None
    assert detect_dialog_event("详细分析一下", updated, _RecordingLogger()).name != "selected_candidate_followup"


@pytest.mark.parametrize("kind,condition,missing,incompatible", CASES)
def test_fallback_allows_ordinary_retrieval_followup(
    monkeypatch, tmp_path, kind, condition, missing, incompatible,
):
    from app import chat_loop as runtime

    paths, facts, _, raw = _drifted_response(kind, condition, missing, incompatible, "missing_candidate")
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return SimpleNamespace(text=raw)
        assert runtime.conversation_state.last_selected_candidate is None
        assert runtime.conversation_state.last_selected_source_files is None
        assert "【已选对象追问】" not in kwargs["contents"]
        return SimpleNamespace(text=f"需要核对{missing}，依据【{paths[0]}】。")

    _run_turns(
        monkeypatch, tmp_path,
        questions=[f"比较这几个{kind}并推荐一个", f"这些{kind}还有哪些信息需要核对？"],
        repo_paths=paths, repo_chunks=facts, state=ConversationState(),
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
    )
    assert len(calls) == 2
    assert missing in runtime.conversation_state.last_answer_text


@pytest.mark.parametrize("invalid", [
    None, 42, [], "", " \n\t", " " * 100_000, "x" * 5_000,
    '{"conclusion": "完整建议", "comparison": "首行来源 03',
    '{"comparison": []}', '{"comparison": "正文", "source_files": "01_甲.md"}',
    '{"comparison": "损坏内容", "comparison": "甲支持离线，依据【01_甲.md】"}',
    '[{"comparison": "依据01_甲.md"}]',
    '```json\n{"comparison": "依据01_甲.md"',
    '{"selected_candidate": "甲", "selected_source_files": ["01_甲.md"]}',
    '{"comparison": "依据臆造来源.md"}',
    "本次生成返回了异常内容，已停止展示，请重试。",
])
def test_invalid_generation_or_structure_preserves_previous_state(
    monkeypatch, tmp_path, capsys, invalid,
):
    from app import chat_loop as runtime

    _run_turns(
        monkeypatch, tmp_path, questions=["比较这几个设备并推荐一个"],
        repo_paths=["01_甲.md", "02_乙.md"], repo_chunks=["合成甲离线", "合成乙联网"],
        state=ConversationState(last_answer_text="上一轮有效回答"),
        client=SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: SimpleNamespace(text=invalid))),
    )
    assert runtime.conversation_state.last_answer_text == "上一轮有效回答"
    assert "请重试" in capsys.readouterr().out
    assert build_comparison_prose_fallback(invalid, ["01_甲.md", "02_乙.md"]) is None


def test_fallback_does_not_rewrite_absolute_claim_or_infer_selected_source():
    raw = json.dumps({
        "comparison": "甲是整体最优，依据【01_甲.md】；乙信息待确认，依据【02_乙.md】。",
        "selected_candidate": "甲", "selected_source_files": ["01_甲.md"],
    }, ensure_ascii=False)
    result = build_comparison_prose_fallback(raw, ["01_甲.md", "02_乙.md"])
    assert "甲是整体最优" in render_decision_result(result)
    assert result.selected_candidate is None
    assert result.selected_source_files == ()
    assert result.source_files == ("01_甲.md", "02_乙.md")


def test_fallback_reuses_footer_evidence_without_promoting_selection_provenance():
    raw = json.dumps({
        "comparison": "甲条件明确；乙仍需确认。", "source_files": ["01_甲.md", "02_乙.md"],
        "selected_candidate": "甲", "selected_source_files": ["03_未参与比较.md"],
    }, ensure_ascii=False)
    result = build_comparison_prose_fallback(raw, ["01_甲.md", "02_乙.md", "03_未参与比较.md"])
    assert result.source_files == ("01_甲.md", "02_乙.md")
    assert "03_未参与比较.md" not in render_decision_result(result)
    assert result.selected_candidate is None


def test_partial_body_without_comparison_field_keeps_available_facts():
    raw = json.dumps({
        "selected_candidate": "甲", "selected_source_files": ["01_甲.md"],
        "reason": "甲支持离线，依据【01_甲.md】；乙必须联网，依据【02_乙.md】。",
        "next_actions": "确认甲的续航时长。",
    }, ensure_ascii=False)
    result = build_comparison_prose_fallback(raw, ["01_甲.md", "02_乙.md"])
    answer = render_decision_result(result)
    assert "甲支持离线" in answer and "乙必须联网" in answer and "确认甲的续航时长" in answer
    assert result.selected_candidate is None
    assert result.source_files == ("01_甲.md", "02_乙.md")


def test_parser_failure_preserves_valid_prose_through_runner(monkeypatch, tmp_path, capsys):
    from app import chat_loop as runtime
    import app.chat_loop_parts.runner as runner

    raw = "甲支持离线，依据【01_甲.md】；乙必须联网，依据【02_乙.md】。"
    monkeypatch.setattr(runner, "parse_decision_result", lambda *_, **__: None)
    _run_turns(
        monkeypatch, tmp_path, questions=["比较这几个设备并推荐一个"],
        repo_paths=["01_甲.md", "02_乙.md"], repo_chunks=["合成甲离线", "合成乙联网"],
        state=ConversationState(),
        client=SimpleNamespace(models=SimpleNamespace(generate_content=lambda **_: SimpleNamespace(text=raw))),
    )
    assert raw in capsys.readouterr().out
    assert runtime.conversation_state.last_selected_candidate is None
    assert runtime.conversation_state.last_selected_source_files is None
    assert runtime.conversation_state.last_answer_source_files == ["01_甲.md", "02_乙.md"]
