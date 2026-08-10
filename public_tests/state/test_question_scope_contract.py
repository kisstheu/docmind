from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.dialog.question_scope import (
    QuestionSignals,
    ScopeDecision,
    analyze_question_signals,
    decide_file_result_set_scope,
)
from app.dialog.state_machine import ConversationState


def _selectable_file_state(
    paths: list[str],
    *,
    focus_file: str | None = None,
) -> ConversationState:
    answer = "\n".join(f"{index}. {path}" for index, path in enumerate(paths, 1))
    return ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_effective_search_query="合成审批主题",
        last_answer_text=answer,
        last_answer_preview=answer,
        last_answer_type="enumeration_file",
        last_result_set_items=list(paths),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
        last_result_set_focus_file=focus_file,
    )


def _decide(
    question: str,
    state: ConversationState,
    *,
    event_name: str,
    current_focus_file: str | None = None,
) -> tuple[QuestionSignals, ScopeDecision]:
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    decision = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=current_focus_file,
        event_name=event_name,
    )
    return signals, decision


@pytest.mark.parametrize(
    ("question", "field"),
    [
        ("Python里的生成器和普通函数有什么区别？", "standalone_general_question"),
        ("这些文件分别讲了什么？", "file_set_content_question"),
        ("它主要解决什么问题？", "explicit_focus_reference"),
        ("第二个文件再详细说说。", "explicit_single_file_result_reference"),
        ("它和第二个文件相比，哪个更适合当前目标？", "result_set_comparison_followup"),
        ("展开说说", "content_followup_question"),
        ("再详细说说。", "detail_explanation_request"),
        ("再总结一下。", "summary_followup_request"),
    ],
)
def test_question_signals_materialize_existing_predicates(question, field):
    signals = analyze_question_signals(
        question,
        last_effective_search_query="合成审批主题",
    )

    assert getattr(signals, field) is True


def test_question_signals_materialize_previous_search_query_dependency():
    question = "请补充相关技术细节说明"

    without_anchor = analyze_question_signals(
        question,
        last_effective_search_query=None,
    )
    with_anchor = analyze_question_signals(
        question,
        last_effective_search_query="合成审批主题",
    )

    assert without_anchor.context_dependent_question is False
    assert with_anchor.context_dependent_question is True


@pytest.mark.parametrize(
    "question",
    [
        "看看某公司A这个合成岗位怎么样。",
        "这份合同符合我们的验收条件吗？",
        "判断一下该采购方案是否满足现有标准。",
    ],
)
def test_document_evaluation_signal_generalizes_across_domains(question):
    signals = analyze_question_signals(
        question,
        last_effective_search_query="合成主题",
    )

    assert signals.document_evaluation_request is True


@pytest.mark.parametrize(
    "question",
    [
        "这份岗位资料列了哪些要求？",
        "总结一下这份合同条款。",
        "第二份采购方案再详细说说。",
    ],
)
def test_document_evaluation_signal_does_not_claim_adjacent_content_requests(question):
    signals = analyze_question_signals(
        question,
        last_effective_search_query="合成主题",
    )

    assert signals.document_evaluation_request is False


def test_question_signals_are_immutable():
    signals = analyze_question_signals(
        "再总结一下。",
        last_effective_search_query="合成审批主题",
    )

    with pytest.raises(FrozenInstanceError):
        signals.summary_followup_request = False


def test_scope_decision_without_selectable_file_result_set():
    state = ConversationState(
        last_result_set_items=["资料甲.md"],
        last_result_set_entity_type="文件",
        last_result_set_selectable=False,
    )

    _signals, decision = _decide(
        "第二个文件再详细说说。",
        state,
        event_name="unknown",
    )

    assert decision.visible_file_paths == ()
    assert decision.file_result_set_selection is None
    assert decision.selected_file_paths is None
    assert decision.result_scope_paths is None


def test_scope_decision_selects_entire_file_set():
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    state = _selectable_file_state(paths)

    _signals, decision = _decide(
        "这些文件分别讲了什么？",
        state,
        event_name="result_set_followup",
    )

    assert decision.visible_file_paths == tuple(paths)
    assert decision.selected_file_paths == tuple(paths)
    assert decision.result_scope_paths == tuple(paths)
    assert decision.query_result_set_items == tuple(paths)
    assert decision.query_result_set_entity == "文件"


def test_scope_decision_selects_explicit_single_ordinal():
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    state = _selectable_file_state(paths)

    _signals, decision = _decide(
        "第二个文件再详细说说。",
        state,
        event_name="result_set_followup",
    )

    assert decision.selected_file_paths == ("记录乙.txt",)
    assert decision.selected_result_set_item_turn is True
    assert decision.requires_result_set_generation is True
    assert decision.result_scope_paths == ("记录乙.txt",)


def test_scope_decision_compares_current_focus_with_ordinal_item():
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    state = _selectable_file_state(paths, focus_file="资料甲.md")

    _signals, decision = _decide(
        "它和第二个文件相比，哪个更适合当前目标？",
        state,
        event_name="result_set_followup",
        current_focus_file="资料甲.md",
    )

    assert decision.selected_file_paths == ("资料甲.md", "记录乙.txt")
    assert decision.result_set_comparison_turn is True
    assert decision.requires_result_set_generation is True
    assert decision.result_scope_paths == ("资料甲.md", "记录乙.txt")


def test_scope_decision_keeps_out_of_range_rejection():
    state = _selectable_file_state(["资料甲.md", "记录乙.txt", "说明丙.pdf"])

    _signals, decision = _decide(
        "第四个文件讲了什么？",
        state,
        event_name="result_set_followup",
    )

    assert decision.file_result_set_selection is not None
    assert decision.file_result_set_selection.paths == ()
    assert (
        decision.file_result_set_selection.rejection
        == "当前结果集中只有 3 个文件，请选择第 1～3 个。"
    )


def test_scope_decision_clears_effective_focus_for_standalone_question():
    paths = ["资料甲.md", "记录乙.txt"]
    state = _selectable_file_state(paths, focus_file="资料甲.md")

    _signals, decision = _decide(
        "Python里的生成器和普通函数有什么区别？",
        state,
        event_name="unknown",
        current_focus_file="资料甲.md",
    )

    assert decision.clear_current_focus is True
    assert decision.effective_focus_file is None
    assert decision.result_scope_paths is None
    assert decision.query_result_set_items is None


def test_scope_decision_content_followup_inherits_single_file_focus():
    paths = ["资料甲.md", "记录乙.txt"]
    state = _selectable_file_state(paths, focus_file="记录乙.txt")

    _signals, decision = _decide(
        "它主要解决什么问题？",
        state,
        event_name="content_followup",
    )

    assert decision.effective_focus_file == "记录乙.txt"
    assert decision.result_scope_paths == ("记录乙.txt",)
    assert decision.query_result_set_items == ("记录乙.txt",)
    assert decision.has_single_focus_scope is True


def test_scope_decision_result_set_followup_keeps_full_query_context():
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    state = _selectable_file_state(paths)

    _signals, decision = _decide(
        "其中哪些最重要？",
        state,
        event_name="result_set_followup",
    )

    assert decision.selected_file_paths is None
    assert decision.result_scope_paths is None
    assert decision.query_result_set_items == tuple(paths)
    assert decision.query_result_set_entity == "文件"


@pytest.mark.parametrize(
    "event_name",
    ["synthesis_request", "structured_request", "structured_skill_summary"],
)
def test_scope_decision_structured_events_keep_full_query_context(event_name):
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    state = _selectable_file_state(paths, focus_file="记录乙.txt")

    _signals, decision = _decide(
        "请按主题归纳",
        state,
        event_name=event_name,
    )

    assert decision.result_scope_paths is None
    assert decision.query_result_set_items == tuple(paths)
    assert decision.query_result_set_entity == "文件"


def test_scope_decision_is_immutable_and_uses_tuples():
    paths = ["资料甲.md", "记录乙.txt"]
    state = _selectable_file_state(paths)
    _signals, decision = _decide(
        "这些文件分别讲了什么？",
        state,
        event_name="result_set_followup",
    )

    assert isinstance(decision.visible_file_paths, tuple)
    assert isinstance(decision.selected_file_paths, tuple)
    assert isinstance(decision.result_scope_paths, tuple)
    assert isinstance(decision.query_result_set_items, tuple)
    with pytest.raises(FrozenInstanceError):
        decision.clear_current_focus = True
