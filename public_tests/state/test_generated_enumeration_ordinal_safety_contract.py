from __future__ import annotations

import pytest

from app.chat_state_helpers import (
    update_state_after_local_answer,
    update_state_after_retrieval_answer,
)
from app.dialog.question_scope import (
    analyze_question_signals,
    decide_file_result_set_scope,
)
from app.dialog.state_machine import ConversationState, detect_dialog_event


class _LoggerStub:
    def debug(self, *_args, **_kwargs):
        return None

    info = debug
    warning = debug
    error = debug


def _file_result_set(paths: list[str]) -> ConversationState:
    answer = "\n".join(f"{index}. {path}" for index, path in enumerate(paths, 1))
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


def _write_generated_followup(
    state: ConversationState,
    *,
    question: str,
    answer: str,
) -> ConversationState:
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
    return update_state_after_retrieval_answer(
        state,
        question,
        answer,
        _LoggerStub(),
        event_name="result_set_followup",
        question_signals=signals,
        scope_decision=scope,
    )


def _numbered_items(prefix: str, count: int) -> str:
    return "\n".join(f"{index}. {prefix}{index}" for index in range(1, count + 1))


def test_real_file_enumeration_keeps_second_file_ordinal_selection():
    paths = ["A.md", "B.md", "C.md"]
    state = _file_result_set(paths)
    question = "第 2 个怎么样？"
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

    assert decision.selected_file_paths == ("B.md",)
    assert decision.file_result_set_selection is not None
    assert decision.file_result_set_selection.rejection is None


def test_repo_meta_file_enumeration_stays_selectable_and_ordered():
    answer = "当前知识库里的文件如下：\n1. A.md\n2. B.md\n3. C.md"

    state = update_state_after_local_answer(
        ConversationState(),
        question="当前有哪些文档？",
        answer=answer,
        route="repo_meta",
        local_topic="list_files",
        is_content_answer=True,
    )

    assert state.last_answer_type == "enumeration_file"
    assert state.last_result_set_items == ["A.md", "B.md", "C.md"]
    assert state.last_result_set_selectable is True


@pytest.mark.parametrize(
    ("old_count", "question", "item_prefix", "visible_count"),
    [
        (14, "有哪些岗位？", "合成岗位", 10),
        (3, "有哪些岗位？", "合成岗位", 3),
        (3, "有哪些审批节点？", "合成节点", 3),
        (4, "有哪些采购方案？", "合成方案", 2),
        (3, "这些文件主要内容是什么？", "合成主题", 3),
    ],
)
def test_generated_numbered_followup_without_mapping_keeps_context_but_disables_ordinal(
    old_count,
    question,
    item_prefix,
    visible_count,
):
    old_files = [f"合成资料{index:02d}.md" for index in range(1, old_count + 1)]
    state = _write_generated_followup(
        _file_result_set(old_files),
        question=question,
        answer=_numbered_items(item_prefix, visible_count),
    )

    assert state.last_result_set_items == old_files
    assert state.last_result_set_entity_type == "文件"
    assert state.last_answer_type is None
    assert state.last_result_set_selectable is False


def test_unsafe_generated_ordinal_returns_local_clarification_without_old_selection():
    old_files = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    state = _write_generated_followup(
        _file_result_set(old_files),
        question="有哪些岗位？",
        answer=_numbered_items("合成岗位", 3),
    )
    question = "第 2 个怎么样？"
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )

    decision = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name="unknown",
    )

    assert decision.selected_file_paths is None
    assert decision.result_scope_paths is None
    assert decision.file_result_set_selection is not None
    assert decision.file_result_set_selection.paths == ()
    assert "无法可靠确定" in (decision.file_result_set_selection.rejection or "")
    assert "明确" in (decision.file_result_set_selection.rejection or "")


def test_explicit_whole_file_set_context_survives_disabled_ordinal_state():
    old_files = ["合同说明.md", "采购记录.md", "验收清单.md"]
    state = _write_generated_followup(
        _file_result_set(old_files),
        question="有哪些审批节点？",
        answer=_numbered_items("合成节点", 3),
    )
    question = "前面那些文件主要内容是什么？"

    event = detect_dialog_event(question, state, _LoggerStub())
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    decision = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name=event.name,
    )

    assert event.name == "result_set_followup"
    assert decision.file_result_set_selection is None
    assert decision.result_scope_paths == tuple(old_files)
    assert decision.query_result_set_items == tuple(old_files)
    assert decision.query_result_set_entity == "文件"


def test_non_file_ordinal_target_does_not_trigger_unsafe_file_rejection():
    state = ConversationState(
        last_result_set_items=["合成资料甲.md", "合成资料乙.md"],
        last_result_set_entity_type="文件",
        last_result_set_selectable=False,
        last_answer_text=_numbered_items("合成节点", 2),
    )
    question = "第二个问题如何解决？"
    signals = analyze_question_signals(
        question,
        last_effective_search_query=None,
    )

    decision = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name="unknown",
    )

    assert decision.file_result_set_selection is None
    assert decision.selected_file_paths is None
