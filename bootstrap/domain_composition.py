from __future__ import annotations

from app.domain_dispatch_port import DomainDispatchPort
from app.domain_host import EmptyDomainHost


def create_domain_host() -> DomainDispatchPort:
    return EmptyDomainHost()
