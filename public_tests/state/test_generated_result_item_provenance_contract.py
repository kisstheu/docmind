from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.chat_state_helpers import update_state_after_retrieval_answer
from app.dialog.question_scope import (
    analyze_question_signals,
    decide_file_result_set_scope,
)
from app.dialog.result_set import materialize_generated_result_set_provenance
from app.dialog.state_machine import ConversationState


class _LoggerStub:
    def debug(self, *_args, **_kwargs):
        return None

    info = debug
    warning = debug
    error = debug


def _repo(documents: dict[str, str]):
    return SimpleNamespace(
        paths=list(documents),
        docs=list(documents.values()),
    )


def _source_file_state(paths: list[str]) -> ConversationState:
    answer = "\n".join(
        f"{index}. {path}" for index, path in enumerate(paths, 1)
    )
    return ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_content_user_question="当前有哪些文档？",
        last_effective_search_query="合成资料集合",
        last_answer_text=answer,
        last_answer_preview=answer,
        last_answer_type="enumeration_file",
        last_result_set_items=list(paths),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
    )


def _generated_state(
    *,
    documents: dict[str, str],
    answer: str,
    question: str = "有哪些条目？",
) -> ConversationState:
    state = _source_file_state(list(documents))
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name="result_set_followup",
    )
    provenance = materialize_generated_result_set_provenance(
        answer,
        candidate_paths=state.last_result_set_items,
        repo_state=_repo(documents),
    )
    return update_state_after_retrieval_answer(
        state,
        question,
        answer,
        _LoggerStub(),
        event_name="result_set_followup",
        question_signals=signals,
        scope_decision=scope,
        generated_result_provenance=provenance,
    )


def _select_generated_ordinal(
    state: ConversationState,
    question: str,
):
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    return decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name="unknown",
    )


def test_unique_generated_item_provenance_selects_backing_not_old_ordinal():
    state = _generated_state(
        documents={
            "A.md": "这里只记录背景信息。",
            "B.md": "标题：岗位甲\n要求：合成能力甲。",
            "C.md": "标题：岗位乙\n要求：合成能力乙。",
        },
        answer="1. 岗位甲\n2. 岗位乙",
        question="有哪些岗位？",
    )

    assert state.last_result_set_items == ["A.md", "B.md", "C.md"]
    assert state.last_result_set_selectable is False
    assert state.last_generated_result_items == ["岗位甲", "岗位乙"]
    assert state.last_generated_result_source_hits == [["B.md"], ["C.md"]]

    decision = _select_generated_ordinal(state, "第1个怎么样？")

    assert decision.selected_file_paths == ("B.md",)
    assert decision.selected_result_set_item_turn is True
    assert decision.result_scope_paths == ("B.md",)
    assert decision.selected_file_paths != ("A.md",)


@pytest.mark.parametrize(
    ("documents", "answer", "ordinal"),
    [
        (
            {"A.md": "背景甲", "B.md": "背景乙", "C.md": "背景丙"},
            "1. 条目甲\n2. 条目乙",
            1,
        ),
        (
            {"A.md": "背景甲", "B.md": "条目甲", "C.md": "条目甲"},
            "1. 条目甲\n2. 条目乙",
            1,
        ),
        (
            {"A.md": "条目甲", "B.md": "条目乙", "C.md": "条目丙"},
            "1. 虚构条目\n2. 条目乙\n3. 条目丙",
            1,
        ),
    ],
    ids=["missing", "ambiguous", "hallucinated"],
)
def test_unproven_generated_item_ordinal_fails_closed(
    documents,
    answer,
    ordinal,
):
    state = _generated_state(documents=documents, answer=answer)

    decision = _select_generated_ordinal(state, f"第{ordinal}个怎么样？")

    assert decision.selected_file_paths is None
    assert decision.result_scope_paths is None
    assert decision.file_result_set_selection is not None
    assert "无法可靠确定" in (
        decision.file_result_set_selection.rejection or ""
    )
    assert state.last_result_set_focus_file is None


def test_equal_generated_and_source_counts_do_not_create_positional_mapping():
    state = _generated_state(
        documents={"A.md": "甲", "B.md": "乙", "C.md": "丙"},
        answer="1. X\n2. Y\n3. Z",
    )

    assert state.last_generated_result_items == ["X", "Y", "Z"]
    assert state.last_generated_result_source_hits == [[], [], []]
    assert _select_generated_ordinal(state, "第2个怎么样？").selected_file_paths is None


def test_partial_generated_result_uses_item_level_selectability():
    state = _generated_state(
        documents={
            "A.md": "条目甲",
            "B.md": "条目乙",
            "C.md": "条目乙",
        },
        answer="1. 条目甲\n2. 条目乙\n3. 条目丙",
    )

    assert state.last_generated_result_source_hits == [
        ["A.md"],
        ["B.md", "C.md"],
        [],
    ]
    assert _select_generated_ordinal(state, "第1个怎么样？").selected_file_paths == (
        "A.md",
    )
    assert _select_generated_ordinal(state, "第2个怎么样？").selected_file_paths is None
    assert _select_generated_ordinal(state, "第3个怎么样？").selected_file_paths is None


def test_provenance_is_source_bounded_and_does_not_use_approximate_text():
    repo_state = _repo(
        {
            "A.md": "项目甲包含明确的里程碑。",
            "outside.md": "项目乙",
            "near.md": "项目乙扩展版",
        }
    )

    provenance = materialize_generated_result_set_provenance(
        "1. 项目乙",
        candidate_paths=["A.md", "near.md"],
        repo_state=repo_state,
    )

    assert provenance.display_items == ("项目乙",)
    assert provenance.source_candidates == ("A.md", "near.md")
    assert provenance.source_hits == ((),)


@pytest.mark.parametrize(
    ("question", "answer", "documents", "expected_backing"),
    [
        (
            "有哪些岗位？",
            "1. 岗位甲\n2. 岗位乙",
            {"A.md": "岗位乙", "B.md": "岗位甲"},
            "B.md",
        ),
        (
            "有哪些付款安排？",
            "1. 分期付款\n2. 验收后付款",
            {"A.md": "验收后付款", "B.md": "分期付款"},
            "B.md",
        ),
        (
            "有哪些项目里程碑？",
            "1. 方案评审\n2. 最终验收",
            {"A.md": "最终验收", "B.md": "方案评审"},
            "B.md",
        ),
    ],
)
def test_unique_provenance_generalizes_across_three_domains(
    question,
    answer,
    documents,
    expected_backing,
):
    state = _generated_state(
        documents=documents,
        answer=answer,
        question=question,
    )

    assert _select_generated_ordinal(state, "第1个怎么样？").selected_file_paths == (
        expected_backing,
    )


def test_real_file_ordinal_still_uses_real_file_result_set():
    state = _source_file_state(["A.md", "B.md", "C.md"])

    decision = _select_generated_ordinal(state, "第1个怎么样？")

    assert decision.selected_file_paths == ("A.md",)


def test_unselectable_generated_state_keeps_source_set_for_non_ordinal_context():
    state = _generated_state(
        documents={"A.md": "背景甲", "B.md": "背景乙", "C.md": "背景丙"},
        answer="1. 条目甲\n2. 条目乙",
    )
    question = "前面那些文件主要内容是什么？"
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )

    decision = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name="result_set_followup",
    )

    assert decision.result_scope_paths == ("A.md", "B.md", "C.md")
    assert decision.query_result_set_items == ("A.md", "B.md", "C.md")
