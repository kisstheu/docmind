from __future__ import annotations

from docmind_domain_sdk import (
    PROTOCOL_VERSION,
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

from .extraction import extract_constraints
from .recognition import recognize_query
from .rendering import render_markdown


PLUGIN_ID = "org.docmind.recruitment.jd-constraints"
PLUGIN_VERSION = "0.1.0"
DISPLAY_NAME = "DocMind Recruitment JD Constraints"


class RecruitmentJDPlugin:
    async def describe(self, request: PluginDescribeRequest) -> PluginManifest:
        return PluginManifest(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            plugin_version=PLUGIN_VERSION,
            schema_version=PROTOCOL_VERSION,
            display_name=DISPLAY_NAME,
            transport_modes=("in_process",),
        )

    async def start(self, request: PluginStartRequest) -> LifecycleResult:
        return LifecycleResult(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            status="ok",
        )

    async def sync_sources(self, request: SourceSyncRequest) -> SourceSyncResult:
        return SourceSyncResult(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            status="ok",
        )

    async def probe(self, request: DomainRequest) -> ProbeResult:
        if recognize_query(request.query) is not None:
            return ProbeResult(
                request_id=request.request_id,
                plugin_id=PLUGIN_ID,
                disposition="claim",
                score=1.0,
                reason_code="recruitment.structured-jd",
            )
        return ProbeResult(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            disposition="abstain",
            score=0.0,
            reason_code="recruitment.unsupported",
        )

    async def execute(self, request: DomainRequest) -> DomainResult:
        extracted = extract_constraints(request.query)
        if extracted is not None:
            return DomainResult(
                request_id=request.request_id,
                plugin_id=PLUGIN_ID,
                status="handled",
                answer_markdown=render_markdown(extracted),
            )
        return DomainResult(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            status="abstain",
        )

    async def stop(self, request: PluginStopRequest) -> LifecycleResult:
        return LifecycleResult(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            status="ok",
        )
