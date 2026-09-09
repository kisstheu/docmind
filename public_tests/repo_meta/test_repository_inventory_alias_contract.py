from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from ai.query_router import route_question
from ai.repo_meta.answering import answer_repo_meta_question
from ai.repo_meta.classifier import classify_repo_meta_question, parse_file_list_request
from app.dialog.state_machine import ConversationState, detect_dialog_event


class _LoggerStub:
    def debug(self, *_args, **_kwargs):
        return None

    info = debug
    warning = debug
    error = debug


@pytest.mark.parametrize(
    "question",
    [
        "有哪些笔记？",
        "当前有哪些笔记？",
        "有哪些文档？",
        "有哪些文件？",
        "有什么资料？",
        "有哪些资料？",
    ],
)
def test_repository_inventory_noun_aliases_use_one_local_file_list_intent(question):
    assert parse_file_list_request(question) == ""
    assert classify_repo_meta_question(question) == "list_files"

    event = detect_dialog_event(question, ConversationState(), _LoggerStub())
    assert event.name == "repo_meta_request"
    assert event.route_hint == "repo_meta"


@pytest.mark.parametrize(
    "question",
    [
        "笔记里提到了什么？",
        "总结一下这篇笔记",
        "关于 Python 的笔记说了什么？",
        "关于面试的笔记有哪些内容？",
    ],
)
def test_note_content_questions_are_not_taken_over_by_repository_inventory(question):
    assert parse_file_list_request(question) is None
    assert classify_repo_meta_question(question) != "list_files"
    assert detect_dialog_event(question, ConversationState(), _LoggerStub()).name != "repo_meta_request"


def test_explicit_note_inventory_does_not_call_local_intent_model(monkeypatch):
    def _unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("explicit repository inventory must not call the local model")

    monkeypatch.setattr("ai.query_router.requests.post", _unexpected_model_call)

    assert route_question(
        "有哪些笔记？",
        "http://127.0.0.1:11434/api/generate",
        "synthetic-model",
        _LoggerStub(),
    ) == {"route": "repo_meta"}


def test_note_inventory_lists_indexed_files_without_document_body_or_models():
    paths = ["合成说明甲.md", "合成记录乙.pdf"]
    repo_state = SimpleNamespace(
        paths=paths,
        all_files=list(paths),
        file_times=[datetime(2026, 1, 1), datetime(2026, 1, 2)],
    )

    answer, topic = answer_repo_meta_question("有哪些笔记？", repo_state)

    assert topic == "list_files"
    assert answer.splitlines() == [
        "当前知识库里的文件如下：",
        "1. 合成说明甲.md",
        "2. 合成记录乙.pdf",
    ]
