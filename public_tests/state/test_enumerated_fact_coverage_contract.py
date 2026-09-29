from __future__ import annotations

from types import SimpleNamespace

import pytest

from ai.enumerated_fact_coverage import (
    assess_enumerated_fact_coverage,
    extract_enumerated_fact_targets,
)
from ai.prompt_builder import build_final_prompt


CASES = [
    (
        "合同甲的签署日期、履行周期和生效阈值分别是多少？",
        ("签署日期", "履行周期", "生效阈值"),
        "签署日期：2042年7月9日；履行周期：18个月；生效阈值：达到80分。",
    ),
    (
        "课程甲的开始时间、持续时长和及格标准分别是多少？",
        ("开始时间", "持续时长", "及格标准"),
        "开始时间：上午9点；持续时长：45分钟；及格标准：72分。",
    ),
    (
        "订单甲的交付时间、包装件数和合格阈值分别是多少？",
        ("交付时间", "包装件数", "合格阈值"),
        "交付时间：确认后8天；包装件数：24件；合格阈值：96%。",
    ),
]


@pytest.mark.parametrize("question,targets,complete_answer", CASES)
def test_enumerated_fact_prompt_materializes_every_requested_field(
    question, targets, complete_answer,
):
    prompt = build_final_prompt([], None, "", "合成参考事实。", question)

    assert extract_enumerated_fact_targets(question) == targets
    assert "【多项事实完整性交付】" in prompt
    assert [prompt.index(f"{index}. {target}") for index, target in enumerate(targets, 1)] == sorted(
        prompt.index(f"{index}. {target}") for index, target in enumerate(targets, 1)
    )
    assert assess_enumerated_fact_coverage(question, complete_answer).complete


@pytest.mark.parametrize("question,targets,complete_answer", CASES)
def test_partial_generated_answer_is_not_complete_when_later_fields_are_missing(
    question, targets, complete_answer,
):
    partial = f"{targets[0]}：已确认。"

    coverage = assess_enumerated_fact_coverage(question, partial)

    assert coverage.complete is False
    assert coverage.missing == targets[1:]


@pytest.mark.parametrize(
    "question",
    [
        "合同甲的签署日期是多少？",
        "比较合同甲与合同乙的履行周期。",
        "列出这些资料中的风险。",
    ],
)
def test_adjacent_non_enumerated_requests_do_not_enable_the_guard(question):
    assert extract_enumerated_fact_targets(question) == ()
    assert assess_enumerated_fact_coverage(question, "任意已有回答。").complete
    assert "【多项事实完整性交付】" not in build_final_prompt(
        [], None, "", "合成参考事实。", question,
    )


def test_explicit_partial_answer_remains_deliverable_when_evidence_is_incomplete():
    question, targets, _ = CASES[2]
    answer = f"{targets[0]}：确认后8天；其他请求项尚需补充证据。"

    coverage = assess_enumerated_fact_coverage(question, answer)

    assert coverage.complete is False
    assert coverage.explicit_partial is True
    assert coverage.safe_to_deliver is True


def test_enumerated_fact_coverage_can_be_rolled_back(monkeypatch):
    monkeypatch.setenv("DOCMIND_ENUMERATED_FACT_COVERAGE", "0")
    question, _, _ = CASES[0]

    assert extract_enumerated_fact_targets(question) == ()
    assert "【多项事实完整性交付】" not in build_final_prompt(
        [], None, "", "合成参考事实。", question,
    )


def test_runner_rejects_a_partial_multifield_draft_before_final_delivery(
    monkeypatch, tmp_path, capsys,
):
    from app import chat_loop as runtime
    from app.chat_loop_parts import runner
    from app.dialog.state_machine import ConversationState
    from public_tests.state.test_repo_meta_generic_file_list_scope_contract import _run_turns

    question, targets, _ = CASES[1]
    partial = f"{targets[0]}：上午9点。"
    state = ConversationState(last_answer_text="上一轮有效回答")
    client = SimpleNamespace(models=SimpleNamespace(
        generate_content=lambda **_: SimpleNamespace(text=partial),
    ))
    monkeypatch.setattr(runner, "maybe_build_direct_lookup_answer", lambda **_: None)

    _run_turns(
        monkeypatch,
        tmp_path,
        questions=[question],
        repo_paths=["合成课程资料.md"],
        repo_chunks=["开始时间为上午9点，持续时长为45分钟，及格标准为72分。"],
        state=state,
        client=client,
    )

    output = capsys.readouterr().out
    assert "已停止展示不完整答案" in output
    assert partial not in output
    assert runtime.conversation_state.last_answer_text == "上一轮有效回答"
