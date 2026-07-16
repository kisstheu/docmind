from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import uuid4

from docmind_domain_sdk import DomainRequest, DomainResult


@runtime_checkable
class DomainDispatchPort(Protocol):
    def dispatch(self, request: DomainRequest) -> DomainResult | None:
        ...


def dispatch_domain_request(
    port: DomainDispatchPort,
    question: str,
) -> DomainResult | None:
    request = DomainRequest(
        request_id=uuid4().hex,
        query=question,
        source_scope=(),
    )
    try:
        result = port.dispatch(request)
    except Exception:
        return None
    return result if isinstance(result, DomainResult) else None
