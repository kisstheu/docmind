from __future__ import annotations

import pytest

from ai.capability_identity import (
    ASSISTANT_IDENTITY_REPLY,
    answer_assistant_identity_question,
    is_assistant_identity_request,
)
from ai.capability_smalltalk import answer_smalltalk
from ai.capability_system import answer_system_capability_question
from ai.query_router import route_question
from ai.query_router_rules import _is_definitely_out_of_scope
from app.chat_loop_handlers import try_handle_assistant_identity
from app.dialog_state_machine import ConversationState, detect_dialog_event
from app.retrieval_flow.routing import resolve_route


class _Logger:
    def debug(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


@pytest.mark.parametrize(
    "question",
    [
        "你谁？",
        "你是谁？",
        "你是啥？",
        "你叫什么？",
        "你的名字是什么？",
        "你是什么助手？",
        "请介绍一下你自己。",
    ],
)
def test_identity_variants_are_answered_by_core_without_provider(monkeypatch, question):
    def unexpected_provider_call(*_args, **_kwargs):
        raise AssertionError("assistant identity must not call a routing or answer provider")

    monkeypatch.setattr("ai.query_router.requests.post", unexpected_provider_call)
    monkeypatch.setattr(
        "app.chat_loop_handlers._answer_smalltalk_with_local_llm",
        unexpected_provider_call,
    )
    monkeypatch.setattr(
        "app.chat_loop_handlers._answer_out_of_scope_with_local_llm",
        unexpected_provider_call,
    )

    state = ConversationState(
        mode="content",
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_user_question="合同里有哪些付款条件？",
        last_content_user_question="合同里有哪些付款条件？",
        last_effective_search_query="合同 付款条件",
        last_answer_text="既有本地资料回答",
    )
    event = detect_dialog_event(question, state, _Logger())
    route_info = resolve_route(
        question,
        event,
        "http://127.0.0.1:9",
        "provider-persona-must-not-run",
        _Logger(),
        state=state,
    )
    answer = try_handle_assistant_identity(route_info["route"], question)

    assert event.name == "assistant_identity"
    assert event.route_hint == "assistant_identity"
    assert route_info["route"] == "assistant_identity"
    assert answer == ASSISTANT_IDENTITY_REPLY
    assert "DocMind" in answer
    assert _is_definitely_out_of_scope(question) is False


@pytest.mark.parametrize(
    "question",
    [
        "你用的什么模型？",
        "你是什么模型？",
        "谁开发的你？",
        "你是哪家公司开发的？",
        "你能做什么？",
        "候选人X是谁？",
        "合同里的助手是谁？",
        "采购系统使用什么模型？",
    ],
)
def test_adjacent_product_and_cross_domain_questions_are_not_identity(question):
    assert is_assistant_identity_request(question) is False
    assert answer_assistant_identity_question(question) is None


def test_existing_neighbor_routes_remain_distinct_without_provider(monkeypatch):
    def unexpected_provider_call(*_args, **_kwargs):
        raise AssertionError("these established rule routes must remain deterministic")

    monkeypatch.setattr("ai.query_router.requests.post", unexpected_provider_call)

    assert route_question(
        "你能做什么？",
        "http://127.0.0.1:9",
        "unused",
        _Logger(),
    )["route"] == "system_capability"
    assert route_question(
        "你几岁？",
        "http://127.0.0.1:9",
        "unused",
        _Logger(),
    )["route"] == "smalltalk"
    assert route_question(
        "你用的什么模型？",
        "http://127.0.0.1:9",
        "unused",
        _Logger(),
    )["route"] == "out_of_scope"
    assert route_question(
        "谁开发的你？",
        "http://127.0.0.1:9",
        "unused",
        _Logger(),
    )["route"] == "out_of_scope"


def test_identity_is_not_owned_by_smalltalk_or_system_capability_answers():
    assert answer_smalltalk("你叫什么？") is None
    assert answer_system_capability_question("你是谁？") is None
    assert answer_assistant_identity_question("你叫什么？") == ASSISTANT_IDENTITY_REPLY
