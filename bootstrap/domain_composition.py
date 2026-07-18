from __future__ import annotations

from docmind_domain_sdk import DomainPlugin

from app.domain_dispatch_port import DomainDispatchPort
from app.domain_host import EmptyDomainHost, StaticDomainHost


def create_domain_host(
    plugin: DomainPlugin | None = None,
    expected_plugin_id: str | None = None,
) -> DomainDispatchPort:
    if plugin is None and expected_plugin_id is None:
        return EmptyDomainHost()
    if plugin is None or expected_plugin_id is None:
        raise ValueError("plugin and expected_plugin_id must be provided together")
    return StaticDomainHost(plugin, expected_plugin_id)
