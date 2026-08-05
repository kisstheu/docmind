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
from .comparison import compare_job_search_rules
from .comparison_contracts import JobSearchRules
from .comparison_rendering import render_job_rule_comparison
from .recognition import recognize_query
from .request_options import (
    RecruitmentOptionsError,
    parse_recruitment_request_options,
)
from .rendering import render_markdown


PLUGIN_ID = "org.docmind.recruitment.jd-constraints"
PLUGIN_VERSION = "0.1.0"
DISPLAY_NAME = "DocMind Recruitment JD Constraints"
_INVALID_OPTIONS_MARKDOWN = """## 求职规则输入无效

本次未执行显式规则比较。请检查结构化求职规则后重试。"""


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
        if extracted is None:
            return DomainResult(
                request_id=request.request_id,
                plugin_id=PLUGIN_ID,
                status="abstain",
            )

        try:
            rules = parse_recruitment_request_options(request.options)
        except RecruitmentOptionsError:
            return DomainResult(
                request_id=request.request_id,
                plugin_id=PLUGIN_ID,
                status="handled",
                answer_markdown=_INVALID_OPTIONS_MARKDOWN,
            )

        if rules is None or rules == JobSearchRules():
            answer_markdown = render_markdown(extracted)
        else:
            comparison = compare_job_search_rules(extracted, rules)
            answer_markdown = render_job_rule_comparison(comparison)
        return DomainResult(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            status="handled",
            answer_markdown=answer_markdown,
        )

    async def stop(self, request: PluginStopRequest) -> LifecycleResult:
        return LifecycleResult(
            request_id=request.request_id,
            plugin_id=PLUGIN_ID,
            status="ok",
        )
