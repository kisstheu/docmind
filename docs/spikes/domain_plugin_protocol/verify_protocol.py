from __future__ import annotations

import asyncio
import hashlib
import json

from pydantic import ValidationError

from protocol import (
    DomainPlugin,
    DomainRequest,
    DomainResult,
    EvidenceLocator,
    EvidenceRef,
    FocusContext,
    FocusUpdate,
    LifecycleResult,
    OpaqueFocus,
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


class NeutralContractPlugin:
    async def describe(self, request: PluginDescribeRequest) -> PluginManifest:
        return PluginManifest(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            plugin_version="0.1.0",
            schema_version="1",
            display_name="Neutral contract fixture",
            transport_modes=("in_process", "mcp"),
            permissions=("source_content", "persistent_storage"),
        )

    async def start(self, request: PluginStartRequest) -> LifecycleResult:
        return LifecycleResult(request_id=request.request_id, plugin_id=PLUGIN_ID, status="ok")

    async def sync_sources(self, request: SourceSyncRequest) -> SourceSyncResult:
        return SourceSyncResult(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            status="ok",
            accepted_source_ids=tuple(item.source.source_id for item in request.upserts),
            cache_revision="fixture-1",
        )

    async def probe(self, request: DomainRequest) -> ProbeResult:
        return ProbeResult(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            disposition="claim",
            score=0.8,
            evidence_source_refs=(request.source_scope[0].source_id,),
            reason_code="fixture.claim",
        )

    async def execute(self, request: DomainRequest) -> DomainResult:
        first = OpaqueFocus(
            plugin_id=PLUGIN_ID,
            opaque_id="item-1",
            display_label="Item A",
            source_refs=(request.source_scope[0].source_id,),
        )
        second = OpaqueFocus(
            plugin_id=PLUGIN_ID,
            opaque_id="item-2",
            display_label="Item B",
            source_refs=(request.source_scope[0].source_id,),
        )
        return DomainResult(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            status="handled",
            answer_markdown="Two verifiable items were found.",
            focus_update=FocusUpdate(
                mode="replace_collection",
                items=(first, second),
                selected=first,
            ),
            evidence=(
                EvidenceRef(
                    evidence_id="evidence-1",
                    source_ref=request.source_scope[0].source_id,
                    source_revision=request.source_scope[0].revision,
                    locator=EvidenceLocator(kind="text_span", start=0, end=17),
                    excerpt="Synthetic content",
                    confidence=0.9,
                ),
            ),
        )

    async def stop(self, request: PluginStopRequest) -> LifecycleResult:
        return LifecycleResult(request_id=request.request_id, plugin_id=PLUGIN_ID, status="ok")


def assert_json_round_trip(model) -> None:
    payload = model.model_dump(mode="json")
    json.dumps(payload, allow_nan=False)
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model


async def verify() -> None:
    source = SourceRef(
        source_id="source-1",
        revision="revision-1",
        display_label="Synthetic source",
        media_type="text/plain",
    )
    snapshot = SourceSnapshot(
        source=source,
        content_sha256=hashlib.sha256(b"Synthetic content").hexdigest(),
        inline_text="Synthetic content",
    )
    sync_request = SourceSyncRequest(request_id="sync-1", upserts=(snapshot,))
    current_focus = OpaqueFocus(
        plugin_id=PLUGIN_ID,
        opaque_id="current-item",
        display_label="Current item",
        source_refs=(source.source_id,),
    )
    request = DomainRequest(
        request_id="query-1",
        query="List the available items.",
        locale="en",
        source_scope=(source,),
        focus=FocusContext(collection=(current_focus,), selected=current_focus),
        deadline_ms=1000,
    )

    plugin = NeutralContractPlugin()
    assert isinstance(plugin, DomainPlugin)

    describe_request = PluginDescribeRequest(request_id="describe-1")
    manifest = await plugin.describe(describe_request)
    validate_manifest_boundary(describe_request, manifest, expected_plugin_id=PLUGIN_ID)
    start_request = PluginStartRequest(
        request_id="start-1",
        host_instance_id="host-1",
        storage_uri="file:///tmp/plugin-fixture",
    )
    started = await plugin.start(start_request)
    validate_lifecycle_boundary(start_request, started, expected_plugin_id=PLUGIN_ID)
    synced = await plugin.sync_sources(sync_request)
    validate_sync_boundary(sync_request, synced, expected_plugin_id=PLUGIN_ID)
    claim = await plugin.probe(request)
    validate_probe_boundary(request, claim, expected_plugin_id=PLUGIN_ID)
    result = await plugin.execute(request)
    validate_result_boundary(request, result, expected_plugin_id=PLUGIN_ID)
    stop_request = PluginStopRequest(request_id="stop-1", host_instance_id="host-1")
    stopped = await plugin.stop(stop_request)
    validate_lifecycle_boundary(stop_request, stopped, expected_plugin_id=PLUGIN_ID)

    for model in (
        describe_request,
        manifest,
        start_request,
        started,
        sync_request,
        synced,
        request,
        claim,
        result,
        stop_request,
        stopped,
    ):
        assert_json_round_trip(model)

    json.dumps(DomainRequest.model_json_schema(), allow_nan=False)
    json.dumps(DomainResult.model_json_schema(), allow_nan=False)

    invalid_payload = request.model_dump(mode="json")
    invalid_payload["internal_object"] = {"must": "be rejected"}
    try:
        DomainRequest.model_validate(invalid_payload)
    except ValidationError:
        pass
    else:
        raise AssertionError("unknown fields were accepted")

    escaped_selected = current_focus.model_copy(update={"source_refs": ("source-outside-scope",)})
    try:
        FocusContext(collection=(current_focus,), selected=escaped_selected)
    except ValidationError:
        pass
    else:
        raise AssertionError("FocusContext accepted a selected focus with altered source refs")

    try:
        FocusUpdate(
            mode="replace_collection",
            items=(current_focus,),
            selected=escaped_selected,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("FocusUpdate accepted a selected focus with altered source refs")

    escaped_result = result.model_copy(
        update={
            "evidence": (
                EvidenceRef(
                    evidence_id="escaped",
                    source_ref="source-outside-scope",
                    source_revision="revision-1",
                    locator=EvidenceLocator(kind="page", page=1),
                ),
            )
        }
    )
    try:
        validate_result_boundary(request, escaped_result, expected_plugin_id=PLUGIN_ID)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-scope evidence was accepted")

    stale_revision_result = result.model_copy(
        update={
            "evidence": (
                EvidenceRef(
                    evidence_id="stale-revision",
                    source_ref=source.source_id,
                    source_revision="stale-revision",
                    locator=EvidenceLocator(kind="text_span", start=0, end=17),
                    excerpt="Synthetic content",
                ),
            )
        }
    )
    try:
        validate_result_boundary(request, stale_revision_result, expected_plugin_id=PLUGIN_ID)
    except ValueError:
        pass
    else:
        raise AssertionError("evidence with a stale source revision was accepted")

    escaped_claim = claim.model_copy(update={"evidence_source_refs": ("source-outside-scope",)})
    try:
        validate_probe_boundary(request, escaped_claim, expected_plugin_id=PLUGIN_ID)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-scope probe evidence was accepted")

    escaped_sync = synced.model_copy(update={"accepted_source_ids": ("source-outside-scope",)})
    try:
        validate_sync_boundary(sync_request, escaped_sync, expected_plugin_id=PLUGIN_ID)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-scope sync result ID was accepted")

    mismatched_sync_request = synced.model_copy(update={"request_id": "sync-other"})
    try:
        validate_sync_boundary(sync_request, mismatched_sync_request, expected_plugin_id=PLUGIN_ID)
    except ValueError:
        pass
    else:
        raise AssertionError("sync result with a mismatched request ID was accepted")

    mismatched_sync_plugin = synced.model_copy(update={"plugin_id": "org.example.other"})
    try:
        validate_sync_boundary(sync_request, mismatched_sync_plugin, expected_plugin_id=PLUGIN_ID)
    except ValueError:
        pass
    else:
        raise AssertionError("sync result with a mismatched plugin ID was accepted")

    overlapping_sync = synced.model_copy(
        update={
            "accepted_source_ids": (source.source_id,),
            "rejected_source_ids": (source.source_id,),
        }
    )
    try:
        validate_sync_boundary(sync_request, overlapping_sync, expected_plugin_id=PLUGIN_ID)
    except ValueError:
        pass
    else:
        raise AssertionError("overlapping accepted and rejected sync IDs were accepted")

    mismatched_lifecycle = started.model_copy(update={"request_id": "start-other"})
    try:
        validate_lifecycle_boundary(start_request, mismatched_lifecycle, expected_plugin_id=PLUGIN_ID)
    except ValueError:
        pass
    else:
        raise AssertionError("lifecycle result with a mismatched request ID was accepted")

    print("domain plugin protocol verification: ok")


if __name__ == "__main__":
    asyncio.run(verify())
