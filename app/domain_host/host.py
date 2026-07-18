from __future__ import annotations

import asyncio

from docmind_domain_sdk import (
    DomainPlugin,
    DomainRequest,
    DomainResult,
    validate_result_boundary,
)

from app.domain_dispatch_port import DomainDispatchPort


class EmptyDomainHost(DomainDispatchPort):
    def dispatch(self, request: DomainRequest) -> DomainResult | None:
        return None


class StaticDomainHost(DomainDispatchPort):
    def __init__(self, plugin: DomainPlugin, expected_plugin_id: str):
        self._plugin = plugin
        self._expected_plugin_id = expected_plugin_id

    def dispatch(self, request: DomainRequest) -> DomainResult | None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "StaticDomainHost cannot dispatch inside a running event loop"
            )

        result = asyncio.run(self._plugin.execute(request))
        validate_result_boundary(
            request,
            result,
            expected_plugin_id=self._expected_plugin_id,
        )
        return result if result.status == "handled" else None
