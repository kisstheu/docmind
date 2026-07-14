from __future__ import annotations

from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


PROTOCOL_VERSION = "1.0"
PluginId = Annotated[str, Field(pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$", min_length=3, max_length=128)]
OpaqueId = Annotated[str, Field(min_length=1, max_length=512)]
SourceId = Annotated[str, Field(min_length=1, max_length=256)]


class WireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class SourceRef(WireModel):
    source_id: SourceId
    revision: Annotated[str, Field(min_length=1, max_length=256)]
    display_label: Annotated[str, Field(min_length=1, max_length=512)]
    media_type: Annotated[str, Field(min_length=1, max_length=128)]


class SourceSnapshot(WireModel):
    source: SourceRef
    content_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
    inline_text: str | None = None
    resource_uri: Annotated[str, Field(min_length=1, max_length=2048)] | None = None

    @model_validator(mode="after")
    def require_one_content_transport(self) -> "SourceSnapshot":
        transports = int(self.inline_text is not None) + int(self.resource_uri is not None)
        if transports != 1:
            raise ValueError("exactly one content transport is required")
        return self


class EvidenceLocator(WireModel):
    """Locate evidence in one exact source revision.

    ``text_span`` uses zero-based Unicode scalar-value offsets into the exact
    decoded text identified by the parent ``EvidenceRef.source_revision``.
    The interval is left-closed and right-open: ``[start, end)``. Offsets are
    not UTF-8 bytes, UTF-16 code units, or grapheme-cluster indices.
    """

    kind: Literal["text_span", "page", "section", "opaque"]
    start: Annotated[int, Field(ge=0)] | None = None
    end: Annotated[int, Field(gt=0)] | None = None
    page: Annotated[int, Field(ge=1)] | None = None
    section_label: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    opaque_locator: Annotated[str, Field(min_length=1, max_length=1024)] | None = None

    @model_validator(mode="after")
    def validate_locator(self) -> "EvidenceLocator":
        if self.kind == "text_span":
            if self.start is None or self.end is None or self.end <= self.start:
                raise ValueError("text_span requires an increasing start/end pair")
            if any(value is not None for value in (self.page, self.section_label, self.opaque_locator)):
                raise ValueError("text_span cannot carry fields from another locator kind")
        elif self.kind == "page" and self.page is None:
            raise ValueError("page locator requires page")
        elif self.kind == "page":
            if any(
                value is not None
                for value in (self.start, self.end, self.section_label, self.opaque_locator)
            ):
                raise ValueError("page cannot carry fields from another locator kind")
        elif self.kind == "section":
            if self.section_label is None:
                raise ValueError("section locator requires section_label")
            if any(value is not None for value in (self.start, self.end, self.page, self.opaque_locator)):
                raise ValueError("section cannot carry fields from another locator kind")
        elif self.kind == "opaque":
            if self.opaque_locator is None:
                raise ValueError("opaque locator requires opaque_locator")
            if any(value is not None for value in (self.start, self.end, self.page, self.section_label)):
                raise ValueError("opaque cannot carry fields from another locator kind")
        return self


class EvidenceRef(WireModel):
    evidence_id: Annotated[str, Field(min_length=1, max_length=256)]
    source_ref: SourceId
    source_revision: Annotated[str, Field(min_length=1, max_length=256)]
    locator: EvidenceLocator
    excerpt: Annotated[str, Field(max_length=2000)] | None = None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None


class OpaqueFocus(WireModel):
    plugin_id: PluginId
    opaque_id: OpaqueId
    display_label: Annotated[str, Field(min_length=1, max_length=512)]
    source_refs: tuple[SourceId, ...] = ()


def _validate_focus_membership(
    collection: tuple[OpaqueFocus, ...],
    selected: OpaqueFocus | None,
) -> None:
    items_by_key: dict[tuple[str, str], OpaqueFocus] = {}
    for item in collection:
        key = (item.plugin_id, item.opaque_id)
        if key in items_by_key:
            raise ValueError("focus collection contains a duplicate logical identity")
        items_by_key[key] = item

    if selected is None:
        return

    selected_key = (selected.plugin_id, selected.opaque_id)
    collection_item = items_by_key.get(selected_key)
    if collection_item is None:
        raise ValueError("selected focus must belong to the current collection")
    if collection_item != selected:
        raise ValueError("selected focus must exactly match its collection item")


class FocusContext(WireModel):
    collection: tuple[OpaqueFocus, ...] = ()
    selected: OpaqueFocus | None = None

    @model_validator(mode="after")
    def selected_focus_must_belong_to_collection(self) -> "FocusContext":
        _validate_focus_membership(self.collection, self.selected)
        return self


class DomainRequest(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: Annotated[str, Field(min_length=1, max_length=128)]
    query: Annotated[str, Field(min_length=1, max_length=16000)]
    locale: Annotated[str, Field(min_length=2, max_length=32)] = "und"
    source_scope: tuple[SourceRef, ...]
    focus: FocusContext = FocusContext()
    deadline_ms: Annotated[int, Field(ge=50, le=300000)] = 10000

    @model_validator(mode="after")
    def validate_focus_scope(self) -> "DomainRequest":
        source_ids = [source.source_id for source in self.source_scope]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_scope contains duplicate source IDs")
        allowed_source_ids = {source.source_id for source in self.source_scope}
        focus_plugin_ids = {item.plugin_id for item in self.focus.collection}
        if len(focus_plugin_ids) > 1:
            raise ValueError("a focus collection cannot mix plugins")
        for item in self.focus.collection:
            if not set(item.source_refs).issubset(allowed_source_ids):
                raise ValueError("focus points outside the request source scope")
        return self


class ProbeResult(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: Annotated[str, Field(min_length=1, max_length=128)]
    plugin_id: PluginId
    disposition: Literal["claim", "abstain"]
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence_source_refs: tuple[SourceId, ...] = ()
    reason_code: Annotated[str, Field(pattern=r"^[a-z0-9_.-]+$", max_length=128)] | None = None

    @model_validator(mode="after")
    def validate_abstention(self) -> "ProbeResult":
        if self.disposition == "abstain" and (self.score != 0.0 or self.evidence_source_refs):
            raise ValueError("abstention must have zero score and no evidence references")
        return self


class FocusUpdate(WireModel):
    mode: Literal["preserve", "replace_collection", "clear"] = "preserve"
    items: tuple[OpaqueFocus, ...] = ()
    selected: OpaqueFocus | None = None

    @model_validator(mode="after")
    def validate_update(self) -> "FocusUpdate":
        if self.mode != "replace_collection" and (self.items or self.selected is not None):
            raise ValueError("only replace_collection can carry focus items")
        if self.mode == "replace_collection" and not self.items:
            raise ValueError("replace_collection requires at least one focus item")
        if self.mode == "replace_collection":
            _validate_focus_membership(self.items, self.selected)
        return self


class PluginError(WireModel):
    code: Annotated[str, Field(pattern=r"^[a-z0-9_.-]+$", min_length=1, max_length=128)]
    message: Annotated[str, Field(min_length=1, max_length=1000)]
    retryable: bool


class DomainResult(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: Annotated[str, Field(min_length=1, max_length=128)]
    plugin_id: PluginId
    status: Literal["handled", "abstain", "retryable_error", "fatal_error"]
    answer_markdown: Annotated[str, Field(max_length=50000)] = ""
    focus_update: FocusUpdate = FocusUpdate()
    evidence: tuple[EvidenceRef, ...] = ()
    warnings: tuple[Annotated[str, Field(max_length=1000)], ...] = ()
    error: PluginError | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> "DomainResult":
        if self.status == "handled":
            if not self.answer_markdown.strip():
                raise ValueError("handled result requires an answer")
            if self.error is not None:
                raise ValueError("handled result cannot carry an error")
            return self

        if self.answer_markdown or self.evidence or self.focus_update.mode != "preserve":
            raise ValueError("non-handled result cannot mutate state or return partial output")
        if self.status == "abstain" and self.error is not None:
            raise ValueError("abstention is not an execution error")
        if self.status.endswith("_error") and self.error is None:
            raise ValueError("error result requires an error payload")
        return self


class SourceSyncRequest(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: Annotated[str, Field(min_length=1, max_length=128)]
    upserts: tuple[SourceSnapshot, ...] = ()
    removed_source_ids: tuple[SourceId, ...] = ()


class SourceSyncResult(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: Annotated[str, Field(min_length=1, max_length=128)]
    plugin_id: PluginId
    status: Literal["ok", "retryable_error", "fatal_error"]
    accepted_source_ids: tuple[SourceId, ...] = ()
    rejected_source_ids: tuple[SourceId, ...] = ()
    cache_revision: Annotated[str, Field(min_length=1, max_length=256)] | None = None
    error: PluginError | None = None

    @model_validator(mode="after")
    def validate_sync_status(self) -> "SourceSyncResult":
        if self.status == "ok" and self.error is not None:
            raise ValueError("successful sync cannot carry an error")
        if self.status != "ok" and self.error is None:
            raise ValueError("failed sync requires an error payload")
        return self


class PluginDescribeRequest(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: Annotated[str, Field(min_length=1, max_length=128)]


class PluginManifest(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: Annotated[str, Field(min_length=1, max_length=128)]
    plugin_id: PluginId
    plugin_version: Annotated[str, Field(min_length=1, max_length=64)]
    schema_version: Annotated[str, Field(min_length=1, max_length=64)]
    display_name: Annotated[str, Field(min_length=1, max_length=256)]
    transport_modes: tuple[Literal["in_process", "mcp"], ...]
    permissions: tuple[
        Literal["source_content", "persistent_storage", "network", "model_access"], ...
    ] = ()


class PluginStartRequest(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: Annotated[str, Field(min_length=1, max_length=128)]
    host_instance_id: Annotated[str, Field(min_length=1, max_length=128)]
    storage_uri: Annotated[str, Field(min_length=1, max_length=2048)] | None = None
    config: dict[str, JsonValue] = Field(default_factory=dict)


class PluginStopRequest(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: Annotated[str, Field(min_length=1, max_length=128)]
    host_instance_id: Annotated[str, Field(min_length=1, max_length=128)]


class LifecycleResult(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: Annotated[str, Field(min_length=1, max_length=128)]
    plugin_id: PluginId
    status: Literal["ok", "retryable_error", "fatal_error"]
    error: PluginError | None = None

    @model_validator(mode="after")
    def validate_lifecycle_status(self) -> "LifecycleResult":
        if self.status == "ok" and self.error is not None:
            raise ValueError("successful lifecycle operation cannot carry an error")
        if self.status != "ok" and self.error is None:
            raise ValueError("failed lifecycle operation requires an error payload")
        return self


@runtime_checkable
class DomainPlugin(Protocol):
    async def describe(self, request: PluginDescribeRequest) -> PluginManifest: ...

    async def start(self, request: PluginStartRequest) -> LifecycleResult: ...

    async def sync_sources(self, request: SourceSyncRequest) -> SourceSyncResult: ...

    async def probe(self, request: DomainRequest) -> ProbeResult: ...

    async def execute(self, request: DomainRequest) -> DomainResult: ...

    async def stop(self, request: PluginStopRequest) -> LifecycleResult: ...


def validate_result_boundary(
    request: DomainRequest,
    result: DomainResult,
    *,
    expected_plugin_id: str,
) -> None:
    if result.request_id != request.request_id:
        raise ValueError("response request_id mismatch")
    if result.plugin_id != expected_plugin_id:
        raise ValueError("response plugin_id mismatch")

    allowed_source_ids = {source.source_id for source in request.source_scope}
    source_revisions = {source.source_id: source.revision for source in request.source_scope}
    for evidence in result.evidence:
        if evidence.source_ref not in allowed_source_ids:
            raise ValueError("evidence points outside the request source scope")
        if evidence.source_revision != source_revisions[evidence.source_ref]:
            raise ValueError("evidence revision does not match the request source revision")

    for focus in result.focus_update.items:
        if focus.plugin_id != expected_plugin_id:
            raise ValueError("focus belongs to a different plugin")
        if not set(focus.source_refs).issubset(allowed_source_ids):
            raise ValueError("focus points outside the request source scope")


def validate_probe_boundary(
    request: DomainRequest,
    result: ProbeResult,
    *,
    expected_plugin_id: str,
) -> None:
    if result.request_id != request.request_id:
        raise ValueError("probe request_id mismatch")
    if result.plugin_id != expected_plugin_id:
        raise ValueError("probe plugin_id mismatch")

    allowed_source_ids = {source.source_id for source in request.source_scope}
    if not set(result.evidence_source_refs).issubset(allowed_source_ids):
        raise ValueError("probe evidence points outside the request source scope")


def validate_sync_boundary(
    request: SourceSyncRequest,
    result: SourceSyncResult,
    *,
    expected_plugin_id: str,
) -> None:
    if result.request_id != request.request_id:
        raise ValueError("sync request_id mismatch")
    if result.plugin_id != expected_plugin_id:
        raise ValueError("sync plugin_id mismatch")

    requested_source_ids = {
        *(snapshot.source.source_id for snapshot in request.upserts),
        *request.removed_source_ids,
    }
    accepted = set(result.accepted_source_ids)
    rejected = set(result.rejected_source_ids)
    if len(accepted) != len(result.accepted_source_ids):
        raise ValueError("accepted source IDs contain duplicates")
    if len(rejected) != len(result.rejected_source_ids):
        raise ValueError("rejected source IDs contain duplicates")
    if not accepted.issubset(requested_source_ids):
        raise ValueError("accepted source ID was not present in the sync request")
    if not rejected.issubset(requested_source_ids):
        raise ValueError("rejected source ID was not present in the sync request")
    if accepted.intersection(rejected):
        raise ValueError("accepted and rejected source IDs overlap")


def validate_manifest_boundary(
    request: PluginDescribeRequest,
    result: PluginManifest,
    *,
    expected_plugin_id: str,
) -> None:
    if result.request_id != request.request_id:
        raise ValueError("describe request_id mismatch")
    if result.plugin_id != expected_plugin_id:
        raise ValueError("describe plugin_id mismatch")


def validate_lifecycle_boundary(
    request: PluginStartRequest | PluginStopRequest,
    result: LifecycleResult,
    *,
    expected_plugin_id: str,
) -> None:
    if result.request_id != request.request_id:
        raise ValueError("lifecycle request_id mismatch")
    if result.plugin_id != expected_plugin_id:
        raise ValueError("lifecycle plugin_id mismatch")
