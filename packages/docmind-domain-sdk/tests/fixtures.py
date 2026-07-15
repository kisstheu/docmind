from __future__ import annotations

import hashlib

from docmind_domain_sdk import (
    DomainRequest,
    DomainResult,
    EvidenceLocator,
    EvidenceRef,
    FocusUpdate,
    OpaqueFocus,
    PluginDescribeRequest,
    PluginManifest,
    SourceRef,
    SourceSnapshot,
    SourceSyncRequest,
    SourceSyncResult,
)


PLUGIN_ID = "org.example.neutral"
INLINE_TEXT = "  Alpha\r\nCafe\u0301\n"


def source_ref() -> SourceRef:
    return SourceRef(
        source_id="source-1",
        revision="revision-1",
        display_label="Synthetic source",
        media_type="text/plain",
    )


def source_snapshot() -> SourceSnapshot:
    return SourceSnapshot(
        source=source_ref(),
        content_sha256=hashlib.sha256(INLINE_TEXT.encode("utf-8")).hexdigest(),
        inline_text=INLINE_TEXT,
    )


def domain_request() -> DomainRequest:
    return DomainRequest(
        request_id="request-1",
        query="Summarize the available evidence.",
        source_scope=(source_ref(),),
    )


def domain_result() -> DomainResult:
    focus = OpaqueFocus(
        plugin_id=PLUGIN_ID,
        opaque_id="item-1",
        display_label="Synthetic item",
        source_refs=("source-1",),
    )
    return DomainResult(
        request_id="request-1",
        plugin_id=PLUGIN_ID,
        status="handled",
        answer_markdown="A supported result.",
        focus_update=FocusUpdate(
            mode="replace_collection",
            items=(focus,),
            selected=focus,
        ),
        evidence=(
            EvidenceRef(
                evidence_id="evidence-1",
                source_ref="source-1",
                source_revision="revision-1",
                locator=EvidenceLocator(kind="text_span", start=0, end=2),
                excerpt="  ",
            ),
        ),
    )


def manifest_pair() -> tuple[PluginDescribeRequest, PluginManifest]:
    request = PluginDescribeRequest(request_id="describe-1")
    result = PluginManifest(
        request_id=request.request_id,
        plugin_id=PLUGIN_ID,
        plugin_version="0.1.0",
        schema_version="1",
        display_name="Neutral fixture",
        transport_modes=("in_process",),
    )
    return request, result


def sync_pair() -> tuple[SourceSyncRequest, SourceSyncResult]:
    request = SourceSyncRequest(request_id="sync-1", upserts=(source_snapshot(),))
    result = SourceSyncResult(
        request_id=request.request_id,
        plugin_id=PLUGIN_ID,
        status="ok",
        accepted_source_ids=("source-1",),
        cache_revision="cache-1",
    )
    return request, result
