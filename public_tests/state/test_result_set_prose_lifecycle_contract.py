from __future__ import annotations

import pytest

from ai.table_presentation import StructuredTable, refine_structured_table
from app.chat_state_helpers import (
    update_state_after_answer_presentation,
    update_state_after_retrieval_answer,
)
from app.dialog.question_scope import (
    analyze_question_signals,
    decide_file_result_set_scope,
)
from app.dialog.result_set import GeneratedResultSetProvenance
from app.dialog.state_machine import ConversationState, detect_dialog_event


class _LoggerStub:
    def debug(self, *_args, **_kwargs):
        return None

    info = debug
    warning = debug


def _file_result_set(paths: list[str], *, focus: str | None = None) -> ConversationState:
    answer = "\n".join(f"{index}. {path}" for index, path in enumerate(paths, 1))
    return ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_content_user_question="有哪些资料？",
        last_effective_search_query="合成资料",
        last_answer_text=answer,
        last_answer_preview=answer,
        last_answer_type="enumeration_file",
        last_result_set_items=list(paths),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
        last_result_set_focus_file=focus,
    )


def _scope(question: str, state: ConversationState):
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    event = detect_dialog_event(
        question,
        state,
        _LoggerStub(),
        focused_file=state.last_result_set_focus_file,
    )
    decision = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=state.last_result_set_focus_file,
        event_name=event.name,
    )
    return signals, event, decision


def _write_prose_followup(
    state: ConversationState,
    *,
    question: str,
    answer: str,
) -> ConversationState:
    signals, event, decision = _scope(question, state)
    assert event.name == "content_followup"
    assert decision.has_single_focus_scope is True
    return update_state_after_retrieval_answer(
        state,
        question,
        answer,
        _LoggerStub(),
        event_name=event.name,
        focused_file=state.last_result_set_focus_file,
        question_signals=signals,
        scope_decision=decision,
    )


def test_file_list_selection_prose_then_later_ordinal_keeps_canonical_authority():
    paths = ["DOC-001_概览.pdf", "DOC-002_Δ课程说明_2026.docx", "DOC-003_练习.md"]
    state = _file_result_set(paths, focus=paths[1])

    _write_prose_followup(
        state,
        question="它主要涉及什么？",
        answer=(
            "正文说明了课程目标与练习安排。\n"
            "（依据：文件【DOC-002_Δ课程说明_2026.docx】chunk #2）"
        ),
    )

    assert state.last_result_set_items == paths
    assert state.last_result_set_entity_type == "文件"
    assert state.last_result_set_selectable is True
    assert state.last_result_set_focus_file == paths[1]
    assert state.last_answer_type is None

    _signals, event, ordinal = _scope("第3个呢？", state)
    assert event.name == "result_set_followup"
    assert ordinal.file_result_set_selection.rejection is None
    assert ordinal.selected_file_paths == (paths[2],)

    _signals, event, comparison = _scope("把这个文件和第3个文件比较一下。", state)
    assert event.name == "decision_request"
    assert comparison.result_scope_paths == (paths[1], paths[2])


@pytest.mark.parametrize(
    ("paths", "question", "answer"),
    [
        (
            ["说明甲.pdf", "说明乙.docx", "说明丙.md"],
            "这个怎么解释？",
            "1. 正文解释（来源标题：《归档_说明乙.docx》）\n依据：文件【说明乙.docx】chunk #1",
        ),
        (
            ["合同甲.pdf", "课程乙.docx", "合同丙.md"],
            "它涉及什么？",
            "正文涉及交付条件（编号 2）。\n来源：课程乙.docx",
        ),
        (
            ["采购需求甲.pdf", "采购清单乙.xlsx", "验收资料丙.md"],
            "这个怎么说明？",
            "正文说明包装和验收。\n（依据：文件【采购清单乙.xlsx】chunk #3）",
        ),
    ],
    ids=["ordinary-document", "contract-course", "procurement"],
)
def test_filename_like_source_text_in_prose_does_not_pollute_file_result_set(
    paths,
    question,
    answer,
):
    state = _file_result_set(paths, focus=paths[1])

    _write_prose_followup(state, question=question, answer=answer)

    assert state.last_result_set_items == paths
    assert state.last_result_set_selectable is True
    assert state.last_result_set_focus_file == paths[1]
    assert state.last_generated_result_items is None


def test_reliable_generated_enumeration_still_replaces_the_parent_file_set():
    paths = ["事项资料甲.md", "事项资料乙.md", "事项资料丙.md"]
    state = _file_result_set(paths, focus=paths[1])
    provenance = GeneratedResultSetProvenance(
        display_items=("事项甲", "事项乙", "事项丙"),
        source_candidates=tuple(paths),
        source_hits=((paths[0],), (paths[1],), (paths[2],)),
        evidence_hits=(("事项甲",), ("事项乙",), ("事项丙",)),
        opaque_focuses=("focus-a", "focus-b", "focus-c"),
        entity_type="事项",
        enumeration_attempted=True,
        reliable=True,
    )
    question = "有哪些事项？"
    answer = "1. 事项甲\n2. 事项乙\n3. 事项丙"
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    decision = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=state.last_result_set_focus_file,
        event_name="result_set_followup",
    )

    update_state_after_retrieval_answer(
        state,
        question,
        answer,
        _LoggerStub(),
        event_name="result_set_followup",
        generated_result_provenance=provenance,
        question_signals=signals,
        scope_decision=decision,
    )

    assert state.last_result_set_items == ["事项甲", "事项乙", "事项丙"]
    assert state.last_result_set_entity_type == "事项"
    assert state.last_result_set_selectable is True
    assert state.last_generated_result_items == ["事项甲", "事项乙", "事项丙"]


def test_sorted_presentation_keeps_visible_ordinal_bound_to_canonical_identity():
    paths = ["采购资料甲.md", "采购资料乙.md", "采购资料丙.md"]
    state = _file_result_set(paths, focus=paths[1])
    table = StructuredTable(
        columns=("文件", "优先级"),
        rows=((paths[0], "2"), (paths[1], "3"), (paths[2], "1")),
    )
    update_state_after_answer_presentation(
        state,
        "整理成表格",
        "本地表格",
        table=table,
    )
    refinement = refine_structured_table(state.current_presentation_table, "按优先级排列")
    update_state_after_answer_presentation(
        state,
        "按优先级排列",
        "排序后的本地表格",
        table=refinement.table,
        options=refinement.options,
    )

    _signals, _event, decision = _scope("第2个呢？", state)

    assert refinement.valid is True
    assert state.last_result_set_items == paths
    assert state.current_presentation_table.row_identities == (
        paths[2],
        paths[0],
        paths[1],
    )
    assert decision.selected_file_paths == (paths[0],)
