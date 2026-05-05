from __future__ import annotations

from app.chat_loop_parts import runner as _runner
from app.chat_loop_parts.input import (
    _flush_pending_tty_input_unix,
    _has_buffered_console_input,
    _merge_user_question_lines,
    _read_fresh_tty_line,
    _read_user_question,
    _should_use_fresh_tty_input,
)
from app.chat_loop_llm import _answer_out_of_scope_with_local_llm, _answer_smalltalk_with_local_llm
from app.chat_retrieval_flow import build_topic_summarizer
from app.dialog_state_machine import ConversationState
import app.chat_loop_handlers as _loop_handlers

build_chat_config = _loop_handlers.build_chat_config
CONTEXTLESS_FOLLOWUP_REPLY = _loop_handlers.CONTEXTLESS_FOLLOWUP_REPLY
answer_smalltalk = _loop_handlers.answer_smalltalk


def try_handle_contextless_followup(*args, **kwargs):
    return _loop_handlers.try_handle_contextless_followup(*args, **kwargs)


def try_handle_system_capability(*args, **kwargs):
    return _loop_handlers.try_handle_system_capability(*args, **kwargs)


def try_handle_repo_meta(*args, **kwargs):
    return _loop_handlers.try_handle_repo_meta(*args, **kwargs)


def try_handle_smalltalk(*args, **kwargs):
    _loop_handlers.answer_smalltalk = answer_smalltalk
    _loop_handlers._answer_smalltalk_with_local_llm = _answer_smalltalk_with_local_llm
    if kwargs.get("conversation_state") is None:
        kwargs["conversation_state"] = conversation_state
    return _loop_handlers.try_handle_smalltalk(*args, **kwargs)


def try_handle_out_of_scope(*args, **kwargs):
    _loop_handlers.answer_smalltalk = answer_smalltalk
    _loop_handlers._answer_out_of_scope_with_local_llm = _answer_out_of_scope_with_local_llm
    if kwargs.get("conversation_state") is None:
        kwargs["conversation_state"] = conversation_state
    return _loop_handlers.try_handle_out_of_scope(*args, **kwargs)


def try_handle_retrieval_force_local_or_empty_context(*args, **kwargs):
    _loop_handlers.build_topic_summarizer = build_topic_summarizer
    try:
        from app.chat_loop_handlers import result_sets as _result_sets

        _result_sets.build_topic_summarizer = build_topic_summarizer
    except Exception:
        pass
    return _loop_handlers.try_handle_retrieval_force_local_or_empty_context(*args, **kwargs)


conversation_state = ConversationState()


def run_chat_loop(*args, **kwargs):
    return _runner.run_chat_loop(*args, **kwargs)
