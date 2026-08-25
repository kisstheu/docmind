from __future__ import annotations

from collections.abc import Mapping

from .dto import JsonValue


REQUEST_CONTEXT_NAMESPACE = "org.docmind.request-context"
QUESTION_INTENT_EVALUATION = "evaluation"

_QUESTION_INTENT_FIELD = "question_intent"
_SUPPORTED_QUESTION_INTENTS = frozenset((QUESTION_INTENT_EVALUATION,))


def with_question_intent(
    options: Mapping[str, JsonValue] | None,
    question_intent: str,
) -> dict[str, JsonValue]:
    """Return request-local options carrying one host-authoritative intent."""
    if (
        not isinstance(question_intent, str)
        or question_intent not in _SUPPORTED_QUESTION_INTENTS
    ):
        raise ValueError("unsupported question intent")
    merged = dict(options or {})
    merged[REQUEST_CONTEXT_NAMESPACE] = {
        _QUESTION_INTENT_FIELD: question_intent,
    }
    return merged


def get_question_intent(options: Mapping[str, JsonValue]) -> str | None:
    """Read a valid request intent without interpreting plugin-owned options."""
    context = options.get(REQUEST_CONTEXT_NAMESPACE)
    if not isinstance(context, Mapping):
        return None
    if set(context) != {_QUESTION_INTENT_FIELD}:
        return None
    question_intent = context.get(_QUESTION_INTENT_FIELD)
    if (
        not isinstance(question_intent, str)
        or question_intent not in _SUPPORTED_QUESTION_INTENTS
    ):
        return None
    return str(question_intent)


__all__ = [
    "QUESTION_INTENT_EVALUATION",
    "REQUEST_CONTEXT_NAMESPACE",
    "get_question_intent",
    "with_question_intent",
]
