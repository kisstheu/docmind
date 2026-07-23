from __future__ import annotations

import pytest

from ai.repo_meta.classifier import classify_repo_meta_question
from app.chat_state_helpers import update_state_after_local_answer
from app.dialog_state_machine import ConversationState, detect_dialog_event


class _LoggerStub:
    def debug(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


def _detect_and_classify(
    question: str,
    state: ConversationState,
    *,
    last_user_question: str | None = None,
    last_local_topic: str | None = None,
):
    event = detect_dialog_event(question, state, _LoggerStub())
    topic = classify_repo_meta_question(
        question,
        last_user_question=last_user_question,
        last_local_topic=last_local_topic,
    )
    return event, topic


def _repo_meta_file_result_state(
    *,
    local_topic: str = "list_files",
    answer_type: str = "enumeration_file",
    entity_type: str = "文件",
    selectable: bool | None = True,
) -> ConversationState:
    return ConversationState(
        last_route="repo_meta",
        last_content_route="repo_meta",
        last_content_user_question="有哪些文档？",
        last_local_topic=local_topic,
        last_answer_text="当前知识库里的文件如下：\n1. 资料甲.md\n2. 记录乙.txt",
        last_answer_type=answer_type,
        last_result_set_query="有哪些文档？",
        last_result_set_items=["资料甲.md", "记录乙.txt"],
        last_result_set_entity_type=entity_type,
        last_result_set_selectable=selectable,
    )


@pytest.mark.parametrize(
    "question",
    [
        "定期保养在什么时间或里程条件下进行？特殊使用条件会带来什么变化？",
        "合同在什么时间到期？",
        "课程每隔多长时间安排一次？",
    ],
    ids=["maintenance-schedule", "contract-due-date", "course-cycle"],
)
def test_business_time_question_does_not_route_to_repo_meta(question):
    event, topic = _detect_and_classify(question, ConversationState())

    assert event.route_hint != "repo_meta"
    assert topic != "time"


def test_document_due_date_remains_content_question():
    question = "这份合同文档什么时候到期？"

    event, topic = _detect_and_classify(question, ConversationState())

    assert event.route_hint != "repo_meta"
    assert topic != "time"


def test_date_ranged_new_notes_route_to_repo_meta():
    question = "找出六月以后新增的笔记。"

    event, topic = _detect_and_classify(question, ConversationState())

    assert event.route_hint == "repo_meta"
    assert topic == "time"


@pytest.mark.parametrize(
    "question",
    [
        "这份文档是什么时间创建的？",
        "最新的文件是什么？",
        "最早的文件是什么？",
    ],
    ids=["document-created-at", "latest-file", "earliest-file"],
)
def test_explicit_file_time_question_routes_to_repo_meta(question):
    event, topic = _detect_and_classify(question, ConversationState())

    assert event.route_hint == "repo_meta"
    assert topic == "time"


def test_fresh_bare_recent_followup_does_not_route_to_repo_meta():
    event, topic = _detect_and_classify("最近的呢？", ConversationState())

    assert event.route_hint != "repo_meta"
    assert topic != "time"


def test_bare_recent_followup_uses_repo_meta_file_result_context():
    state = _repo_meta_file_result_state()

    event, topic = _detect_and_classify(
        "最近的呢？",
        state,
        last_user_question=state.last_content_user_question,
        last_local_topic=state.last_local_topic,
    )

    assert event.route_hint == "repo_meta"
    assert topic == "time"


@pytest.mark.parametrize(
    "question",
    ["最近的有哪些？", "最近时间有哪些？"],
    ids=["recent-which", "recent-time-which"],
)
def test_existing_repo_meta_recent_followup_routes_stay_green(question):
    state = _repo_meta_file_result_state()

    event, topic = _detect_and_classify(
        question,
        state,
        last_user_question=state.last_content_user_question,
        last_local_topic=state.last_local_topic,
    )

    assert event.route_hint == "repo_meta"
    assert topic == "time"


@pytest.mark.parametrize(
    "question",
    ["最近的有哪些？", "最近时间有哪些？"],
    ids=["recent-which", "recent-time-which"],
)
def test_fresh_recent_list_question_requires_file_result_context(question):
    event, topic = _detect_and_classify(question, ConversationState())

    assert event.route_hint != "repo_meta"
    assert topic != "time"


def test_explicit_recent_file_list_routes_to_repo_meta_time():
    question = "最近时间有哪些文件？"

    event, topic = _detect_and_classify(question, ConversationState())

    assert event.route_hint == "repo_meta"
    assert topic == "time"


@pytest.mark.parametrize(
    "question",
    ["最近更新合同了吗？", "最近资料显示合同到期了吗？"],
    ids=["recent-update-content", "recent-source-content"],
)
def test_business_recent_wording_does_not_use_generic_repo_meta_bypass(question):
    event, topic = _detect_and_classify(question, ConversationState())

    assert event.route_hint != "repo_meta"
    assert topic != "time"


def test_time_result_sequence_keeps_classifier_and_state_machine_aligned():
    state = ConversationState()
    update_state_after_local_answer(
        state,
        "最新的文件是什么？",
        "最新的 2 个文件是：\n1. 资料甲.md（2026-07-22 10:00:00）\n2. 记录乙.txt（2026-07-21 10:00:00）",
        "repo_meta",
        "time",
        True,
    )

    assert state.last_local_topic == "time"
    assert state.last_answer_type == "enumeration_file"
    assert state.last_result_set_items == ["资料甲.md", "记录乙.txt"]
    assert state.last_result_set_entity_type == "文件"
    assert state.last_result_set_selectable is True

    event, topic = _detect_and_classify(
        "最近的呢？",
        state,
        last_user_question=state.last_content_user_question,
        last_local_topic=state.last_local_topic,
    )

    assert event.route_hint == "repo_meta"
    assert topic == "time"


@pytest.mark.parametrize(
    "local_topic",
    ["list_files", "list_files_with_time", "list_files_by_topic", "time"],
)
def test_canonical_file_result_topics_support_bare_recent_followup(local_topic):
    state = _repo_meta_file_result_state(local_topic=local_topic)

    event, topic = _detect_and_classify(
        "最近的呢？",
        state,
        last_user_question=state.last_content_user_question,
        last_local_topic=state.last_local_topic,
    )

    assert event.route_hint == "repo_meta"
    assert topic == "time"


def test_stale_file_items_do_not_restore_context_after_non_file_topic():
    state = _repo_meta_file_result_state(local_topic="count")

    event, topic = _detect_and_classify(
        "最近的呢？",
        state,
        last_user_question=state.last_content_user_question,
        last_local_topic=state.last_local_topic,
    )

    assert event.route_hint != "repo_meta"
    assert topic != "time"


@pytest.mark.parametrize("selectable", [False, None], ids=["false", "none"])
def test_unselectable_file_result_is_not_context(selectable):
    state = _repo_meta_file_result_state(selectable=selectable)

    event = detect_dialog_event("最近的呢？", state, _LoggerStub())

    assert event.route_hint != "repo_meta"


def test_non_file_result_set_is_not_file_time_context():
    state = _repo_meta_file_result_state(entity_type="公司")

    event = detect_dialog_event("最近的呢？", state, _LoggerStub())

    assert event.route_hint != "repo_meta"


@pytest.mark.parametrize(
    "question",
    ["哪份文件是昨天创建的？", "文件创建于哪天？"],
    ids=["created-yesterday", "created-which-day"],
)
def test_explicit_file_creation_time_variants_route_to_repo_meta(question):
    event, topic = _detect_and_classify(question, ConversationState())

    assert event.route_hint == "repo_meta"
    assert topic == "time"
