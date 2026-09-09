from __future__ import annotations

from app.chat_state_answer_parsing import extract_file_items, infer_answer_type
from app.chat_state_helpers import update_state_after_retrieval_answer
from app.chat_text.file_lookup import looks_like_bare_content_question
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


def _active_file_state(paths: list[str]) -> ConversationState:
    answer = "\n".join(f"{index}. {path}" for index, path in enumerate(paths, 1))
    return ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_content_user_question="有哪些合成资料？",
        last_effective_search_query="合成资料",
        last_answer_text=answer,
        last_answer_preview=answer,
        last_answer_type="enumeration_file",
        last_result_set_items=list(paths),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
    )


def _write_retrieval_answer(
    state: ConversationState,
    *,
    question: str,
    answer: str,
    event_name: str,
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
        event_name=event_name,
    )
    return update_state_after_retrieval_answer(
        state,
        question,
        answer,
        _LoggerStub(),
        event_name=event_name,
        question_signals=signals,
        scope_decision=scope,
    )


def test_empty_file_enumeration_extraction_preserves_active_set_without_invalidation():
    paths = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    question = "可以按资料名称排列吗？"
    answer = "以下文件已经排列：\n1. 第一项\n2. 第二项\n3. 第三项"

    assert infer_answer_type(question, answer) == "enumeration_file"
    assert extract_file_items(answer) == []

    state = _write_retrieval_answer(
        _active_file_state(paths),
        question=question,
        answer=answer,
        event_name="content_followup",
    )

    assert state.last_result_set_items == paths
    assert state.last_result_set_entity_type == "文件"
    assert state.last_result_set_selectable is True


def test_content_followup_after_empty_extraction_still_uses_original_file_scope():
    paths = ["合成记录甲.md", "合成记录乙.md", "合成记录丙.md"]
    state = _write_retrieval_answer(
        _active_file_state(paths),
        question="可以按记录名称排列吗？",
        answer="以下文件已经排列：\n1. 第一项\n2. 第二项\n3. 第三项",
        event_name="content_followup",
    )

    question = "具体讲了什么？"
    event = detect_dialog_event(question, state, _LoggerStub())
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name=event.name,
    )

    assert scope.result_scope_paths == tuple(paths)
    assert scope.query_result_set_items == tuple(paths)
    assert scope.query_result_set_entity == "文件"


def test_specific_bare_content_wording_is_domain_neutral_and_subject_safe():
    assert looks_like_bare_content_question("具体讲了什么？") is True
    assert looks_like_bare_content_question("具体说了什么？") is True
    assert looks_like_bare_content_question("具体记录了什么？") is True
    assert looks_like_bare_content_question("关于合成主题具体讲了什么？") is False


def test_valid_new_file_result_set_still_replaces_the_active_set():
    old_paths = ["旧资料甲.md", "旧资料乙.md"]
    new_paths = ["新资料甲.md", "新资料乙.md"]

    state = _write_retrieval_answer(
        _active_file_state(old_paths),
        question="新的主题有哪些文件？",
        answer="以下文件匹配新主题：\n1. 新资料甲.md\n2. 新资料乙.md",
        event_name="content",
    )

    assert state.last_result_set_items == new_paths
    assert state.last_result_set_entity_type == "文件"
    assert state.last_result_set_selectable is True


def test_explicit_standalone_scope_switch_does_not_preserve_on_empty_extraction():
    state = _active_file_state(["旧资料甲.md", "旧资料乙.md"])
    question = "新的主题有哪些文件？"
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    assert signals.standalone_general_question is True

    state = _write_retrieval_answer(
        state,
        question=question,
        answer="以下文件匹配新主题：\n1. 第一项\n2. 第二项",
        event_name="content",
    )

    assert state.last_result_set_items is None
    assert state.last_result_set_entity_type is None
    assert state.last_result_set_selectable is False
