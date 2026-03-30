from __future__ import annotations

from datetime import datetime

from ai.query_router import route_question
from ai.repo_meta.answering import _answer_time
from ai.repo_meta.classifier import classify_repo_meta_question
from app.dialog_state_machine import ConversationState, detect_dialog_event


class _DummyLogger:
    def debug(self, _msg: str):
        return None

    def info(self, _msg: str):
        return None

    def warning(self, _msg: str):
        return None


def test_time_followup_recent_generic_should_map_to_time():
    topic = classify_repo_meta_question(
        "最近的有哪些？",
        last_local_topic="category_summary",
    )
    assert topic == "time"


def test_time_followup_with_time_word_should_map_to_time():
    topic = classify_repo_meta_question("最近时间有哪些？")
    assert topic == "time"


def test_recent_time_files_should_not_be_topic_list():
    topic = classify_repo_meta_question("最近时间有哪些文件？")
    assert topic == "time"


def test_topic_list_with_recent_topic_should_keep_topic_intent():
    topic = classify_repo_meta_question("京东最近有哪些文件？")
    assert topic == "list_files_by_topic"


def test_recent_time_answer_should_default_to_latest_section():
    answer, topic = _answer_time(
        "最近时间有哪些文件？",
        paths=["a.md", "b.md"],
        file_times=[datetime(2026, 3, 20, 10, 0, 0), datetime(2026, 3, 25, 9, 0, 0)],
    )
    assert topic == "time"
    assert "最早" not in answer
    assert answer.splitlines()[1].startswith("1. b.md")


def test_explicit_date_file_question_should_map_to_time():
    topic = classify_repo_meta_question("3月25号还有其他文件吗？")
    assert topic == "time"


def test_answer_time_should_filter_by_explicit_date():
    answer, topic = _answer_time(
        "3月25号还有其他文件吗？",
        paths=["a.md", "b.md", "c.md"],
        file_times=[
            datetime(2026, 3, 25, 9, 0, 0),
            datetime(2026, 3, 25, 17, 51, 44),
            datetime(2026, 3, 26, 8, 0, 0),
        ],
    )
    assert topic == "time"
    assert "3月25日" in answer
    assert "a.md" in answer
    assert "b.md" in answer
    assert "c.md" not in answer


def test_dialog_event_explicit_date_file_followup_should_route_repo_meta():
    state = ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_content_user_question="找下公司名",
    )
    event = detect_dialog_event("3月25号还有其他文件吗？", state, _DummyLogger())
    assert event.name == "repo_meta_request"
    assert event.route_hint == "repo_meta"


def test_short_find_company_name_should_route_to_normal_retrieval_action():
    state = ConversationState(
        last_route="repo_meta",
        last_local_topic="list_files_by_topic",
        last_answer_text="按语义上最接近的类别……",
    )
    event = detect_dialog_event("找下公司名", state, _DummyLogger())
    assert event.name == "action_request"
    assert event.route_hint == "normal_retrieval"


def test_router_should_treat_company_name_lookup_as_normal_retrieval():
    route = route_question("找下公司名", "http://127.0.0.1:11434/api/generate", "qwen", _DummyLogger())
    assert route["route"] == "normal_retrieval"
