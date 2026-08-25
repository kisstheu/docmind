from __future__ import annotations

from collections.abc import Sequence

from docmind_domain_sdk import (
    PROTOCOL_VERSION,
    QUESTION_INTENT_EVALUATION,
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
    get_question_intent,
)

from .extraction import extract_constraints
from .comparison import compare_job_search_rules
from .comparison_contracts import JobSearchRules
from .comparison_rendering import render_job_rule_comparison
from .recognition import DUTY_LABELS, REQUIREMENT_LABELS, recognize_query
from .request_options import (
    RecruitmentOptionsError,
    parse_recruitment_request_options,
)
from .rendering import render_markdown
from .evaluation_rendering import render_unpersonalized_evaluation


PLUGIN_ID = "org.docmind.recruitment.jd-constraints"
PLUGIN_VERSION = "0.1.0"
DISPLAY_NAME = "DocMind Recruitment JD Constraints"
_INVALID_OPTIONS_MARKDOWN = """## 求职规则输入无效

本次未执行显式规则比较。请检查结构化求职规则后重试。"""


class RecruitmentJDPlugin:
    def adapt_content_query(
        self,
        *,
        question: str,
        content_target: str,
        source_term_groups: Sequence[Sequence[str]],
    ) -> str | None:
        """Adapt a listing term only when one source proves recruitment structure."""
        target = "".join((content_target or "").casefold().split())
        if target != "jd":
            return None

        normalized_question = "".join((question or "").casefold().split())
        listing_markers = ("有哪些", "有哪", "有什么", "有啥", "列出", "列下", "盘点")
        if not any(marker in normalized_question for marker in listing_markers):
            return None

        for source_terms in source_term_groups:
            normalized_terms = {str(term or "").strip() for term in source_terms}
            has_duty_structure = any(label in normalized_terms for label in DUTY_LABELS)
            has_requirement_structure = any(
                label in normalized_terms for label in REQUIREMENT_LABELS
            )
            if has_duty_structure and has_requirement_structure:
                return "岗位"
        return None

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
            if get_question_intent(request.options) == QUESTION_INTENT_EVALUATION:
                answer_markdown = render_unpersonalized_evaluation(extracted)
            else:
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
