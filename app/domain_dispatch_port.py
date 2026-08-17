from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable
from uuid import uuid4

from docmind_domain_sdk import DomainRequest, DomainResult, JsonValue


@runtime_checkable
class DomainDispatchPort(Protocol):
    def dispatch(self, request: DomainRequest) -> DomainResult | None:
        ...


def adapt_domain_content_query(
    port: DomainDispatchPort,
    *,
    question: str,
    content_target: str,
    source_term_groups: Sequence[Sequence[str]],
) -> str | None:
    adapter = getattr(port, "adapt_content_query", None)
    if not callable(adapter):
        return None
    try:
        adapted = adapter(
            question=question,
            content_target=content_target,
            source_term_groups=source_term_groups,
        )
    except Exception:
        return None
    if not isinstance(adapted, str):
        return None
    normalized = adapted.strip()
    return normalized or None


def dispatch_domain_request(
    port: DomainDispatchPort,
    question: str,
    *,
    options: Mapping[str, JsonValue] | None = None,
) -> DomainResult | None:
    request_options = {} if options is None else {"options": options}
    request = DomainRequest(
        request_id=uuid4().hex,
        query=question,
        source_scope=(),
        **request_options,
    )
    try:
        result = port.dispatch(request)
    except Exception:
        return None
    return result if isinstance(result, DomainResult) else None
