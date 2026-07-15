from __future__ import annotations

from typing import Protocol, runtime_checkable

from .dto import (
    DomainRequest,
    DomainResult,
    LifecycleResult,
    PluginDescribeRequest,
    PluginManifest,
    PluginStartRequest,
    PluginStopRequest,
    ProbeResult,
    SourceSyncRequest,
    SourceSyncResult,
)


@runtime_checkable
class DomainPlugin(Protocol):
    async def describe(self, request: PluginDescribeRequest) -> PluginManifest: ...

    async def start(self, request: PluginStartRequest) -> LifecycleResult: ...

    async def sync_sources(self, request: SourceSyncRequest) -> SourceSyncResult: ...

    async def probe(self, request: DomainRequest) -> ProbeResult: ...

    async def execute(self, request: DomainRequest) -> DomainResult: ...

    async def stop(self, request: PluginStopRequest) -> LifecycleResult: ...
