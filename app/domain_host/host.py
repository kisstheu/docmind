from __future__ import annotations

from docmind_domain_sdk import DomainRequest, DomainResult

from app.domain_dispatch_port import DomainDispatchPort


class EmptyDomainHost(DomainDispatchPort):
    def dispatch(self, request: DomainRequest) -> DomainResult | None:
        return None
