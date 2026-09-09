from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ai.generation_output import (
    MAX_GENERATED_OUTPUT_CHARS,
    MAX_REPEATED_CHARACTER_RUN,
    enforce_bounded_absence_claims,
    validate_generated_output,
)


@pytest.mark.parametrize(
    "text",
    [
        "这是正常回答。",
        "| 项目 | 说明 |\n|---|---|\n| 合成项甲 | 正常内容 |",
    ],
)
def test_normal_generated_text_is_accepted_and_trimmed(text):
    result = validate_generated_output(f"\n{text}\n")

    assert result.valid is True
    assert result.reason is None
    assert result.text == text


@pytest.mark.parametrize("text", ["", " \n\t" * 10_000])
def test_empty_or_whitespace_only_generated_text_is_rejected(text):
    result = validate_generated_output(text)

    assert result.valid is False
    assert result.reason == "empty_after_strip"
    assert result.text == ""


def test_large_whitespace_dominant_generated_text_is_rejected():
    result = validate_generated_output("有效开头" + " " * 100_000)

    assert result.valid is False
    assert result.reason == "whitespace_dominance"


def test_unbounded_repeated_character_output_is_rejected():
    result = validate_generated_output("异常" + "x" * MAX_REPEATED_CHARACTER_RUN)

    assert result.valid is False
    assert result.reason == "repeated_character_run"


def test_reasonably_long_normal_answer_is_not_rejected():
    text = "\n".join(
        f"第{index}项：这是领域中立的正常长回答内容，包含不同编号。"
        for index in range(4_000)
    )
    assert len(text) < MAX_GENERATED_OUTPUT_CHARS

    result = validate_generated_output(text)

    assert result.valid is True
    assert result.text == text


def test_global_size_ceiling_rejects_an_unbounded_response():
    text = "正常但无界的生成段落。" * (MAX_GENERATED_OUTPUT_CHARS // 10 + 1)
    assert len(text) > MAX_GENERATED_OUTPUT_CHARS

    result = validate_generated_output(text)

    assert result.valid is False
    assert result.reason == "size_limit"


@pytest.mark.parametrize("entity_type", ["岗位", "条款", "设备"])
@pytest.mark.parametrize("response_kind", ["normal", "whitespace", "oversized", "parsed_only"])
def test_structured_enumeration_respects_output_validation_before_binding(
    monkeypatch, tmp_path, capsys, entity_type, response_kind,
):
    from app import chat_loop as runtime
    import app.chat_loop_parts.runner as runner
    from public_tests.state.test_repo_meta_generic_file_list_scope_contract import (
        _canonical_source_id_from_prompt,
        _run_turns,
        _selectable_file_state,
    )

    paths = ["资料甲.md", "资料乙.md"]
    state = _selectable_file_state(paths)
    previous_answer = state.last_answer_text
    display_name = f"合成{entity_type}甲"
    calls = []
    bound_payloads = []
    real_materialize = runner.materialize_structured_generated_result_set

    def capture_binding(payload, **kwargs):
        bound_payloads.append(payload)
        return real_materialize(payload, **kwargs)

    def generate_content(*, model, contents, config=None):
        calls.append(contents)
        payload = {"items": [{
            "display_name": display_name,
            "source_ids": [_canonical_source_id_from_prompt(contents, paths[0])],
            "evidence_text": display_name,
        }]}
        text = json.dumps(payload, ensure_ascii=False)
        if response_kind == "whitespace":
            text += " " * 100_000
        elif response_kind == "oversized":
            text += " " * MAX_GENERATED_OUTPUT_CHARS
        elif response_kind == "parsed_only":
            text = None
        return SimpleNamespace(text=text, parsed=payload)

    monkeypatch.setattr(runner, "materialize_structured_generated_result_set", capture_binding)
    _run_turns(
        monkeypatch, tmp_path,
        questions=[f"总结一下有哪些{entity_type}？"],
        repo_paths=paths,
        repo_chunks=[f"名称：{display_name}。", "这里只记录合成背景。"],
        state=state,
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
    )

    output = capsys.readouterr().out
    assert len(calls) == 1
    assert len(bound_payloads) == 1
    updated = runtime.conversation_state
    if response_kind == "normal":
        assert bound_payloads[0] is not None
        assert updated.last_generated_result_items == [display_name]
        assert f"1. {display_name}" in output
    else:
        assert bound_payloads == [None]
        assert updated.last_answer_text == previous_answer
        assert updated.last_result_set_items == paths
        assert updated.last_generated_result_items is None
        assert "本轮未生成可可靠引用的枚举结果，请重试。" in output
        assert f"1. {display_name}" not in output


@pytest.mark.parametrize(
    "text",
    [
        "资料未提及该岗位的年龄要求。",
        "文件【合同资料.md】没有规定该事项的期限。",
        "材料中不存在该设备的保修期。",
        "文档未明确说明该课程的先修条件。",
    ],
)
def test_limited_retrieval_source_absence_claim_is_bounded(text):
    bounded = enforce_bounded_absence_claims(text)

    assert "当前检索到的" in bounded
    assert "证据中暂未找到明确说明" in bounded
    assert not any(
        phrase in bounded
        for phrase in ("资料未提及", "没有规定", "材料中不存在", "文档未明确说明")
    )


@pytest.mark.parametrize(
    "text",
    [
        "当前检索到的证据中暂未找到明确说明。",
        "条款明确规定仅适用于合成范围甲，不包括合成范围乙。",
        "来源明确写明该条件不适用。",
    ],
)
def test_explicit_negative_or_already_bounded_wording_is_not_rewritten(text):
    assert enforce_bounded_absence_claims(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "该岗位常见于资深人员，因此不适用于初级人员。",
        "该条款列出了长期事项，所以不适用于短期事项。",
        "该设备通常用于大型项目，不适用于小型项目。",
        "该课程面向高年级学习者，由此可见不适用于低年级学习者。",
    ],
)
def test_inferred_exclusion_without_explicit_boundary_is_bounded(text):
    bounded = enforce_bounded_absence_claims(text)

    assert "当前检索到的证据不足以确认对" in bounded
    assert bounded != text


@pytest.mark.parametrize(
    "text",
    [
        "条款明确规定仅适用于长期事项，因此不适用于短期事项。",
        "原文明确写明该设备不适用于小型项目。",
        "参考片段明确说明该课程仅限高年级学习者，不适用于低年级学习者。",
    ],
)
def test_explicit_boundary_evidence_preserves_negative_applicability(text):
    assert enforce_bounded_absence_claims(text) == text
