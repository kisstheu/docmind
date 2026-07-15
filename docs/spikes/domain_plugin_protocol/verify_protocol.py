from __future__ import annotations

import asyncio
import hashlib

from docmind_domain_sdk import (
    DomainPlugin,
    DomainRequest,
    DomainResult,
    LifecycleResult,
    PluginDescribeRequest,
    PluginManifest,
    PluginStartRequest,
    PluginStopRequest,
    ProbeResult,
    SourceRef,
    SourceSnapshot,
    SourceSyncRequest,
    SourceSyncResult,
    validate_lifecycle_boundary,
    validate_manifest_boundary,
    validate_probe_boundary,
    validate_result_boundary,
    validate_sync_boundary,
)


PLUGIN_ID = "org.example.neutral"


class NeutralSmokePlugin:
    async def describe(self, request: PluginDescribeRequest) -> PluginManifest:
        return PluginManifest(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            plugin_version="0.1.0",
            schema_version="1",
            display_name="Neutral smoke fixture",
            transport_modes=("in_process",),
        )

    async def start(self, request: PluginStartRequest) -> LifecycleResult:
        return LifecycleResult(request_id=request.request_id, plugin_id=PLUGIN_ID, status="ok")

    async def sync_sources(self, request: SourceSyncRequest) -> SourceSyncResult:
        return SourceSyncResult(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            status="ok",
            accepted_source_ids=tuple(item.source.source_id for item in request.upserts),
        )

    async def probe(self, request: DomainRequest) -> ProbeResult:
        return ProbeResult(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            disposition="claim",
            score=0.5,
            evidence_source_refs=(request.source_scope[0].source_id,),
        )

    async def execute(self, request: DomainRequest) -> DomainResult:
        return DomainResult(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            status="handled",
            answer_markdown="A synthetic result.",
        )

    async def stop(self, request: PluginStopRequest) -> LifecycleResult:
        return LifecycleResult(request_id=request.request_id, plugin_id=PLUGIN_ID, status="ok")


async def verify() -> None:
    source = SourceRef(
        source_id="source-1",
        revision="revision-1",
        display_label="Synthetic source",
        media_type="text/plain",
    )
    text = "Synthetic content"
    sync_request = SourceSyncRequest(
        request_id="sync-1",
        upserts=(
            SourceSnapshot(
                source=source,
                content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                inline_text=text,
            ),
        ),
    )
    request = DomainRequest(
        request_id="request-1",
        query="Summarize the source.",
        source_scope=(source,),
    )
    plugin = NeutralSmokePlugin()
    assert isinstance(plugin, DomainPlugin)

    describe_request = PluginDescribeRequest(request_id="describe-1")
    manifest = await plugin.describe(describe_request)
    validate_manifest_boundary(describe_request, manifest, expected_plugin_id=PLUGIN_ID)
    start_request = PluginStartRequest(request_id="start-1", host_instance_id="host-1")
    started = await plugin.start(start_request)
    validate_lifecycle_boundary(start_request, started, expected_plugin_id=PLUGIN_ID)
    synced = await plugin.sync_sources(sync_request)
    validate_sync_boundary(sync_request, synced, expected_plugin_id=PLUGIN_ID)
    probe = await plugin.probe(request)
    validate_probe_boundary(request, probe, expected_plugin_id=PLUGIN_ID)
    result = await plugin.execute(request)
    validate_result_boundary(request, result, expected_plugin_id=PLUGIN_ID)
    stop_request = PluginStopRequest(request_id="stop-1", host_instance_id="host-1")
    stopped = await plugin.stop(stop_request)
    validate_lifecycle_boundary(stop_request, stopped, expected_plugin_id=PLUGIN_ID)
    print("domain SDK protocol 1.0 smoke: ok")


if __name__ == "__main__":
    asyncio.run(verify())
