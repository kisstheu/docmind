from __future__ import annotations

import hashlib
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)


PROTOCOL_VERSION = "1.0"


def _reject_surrounding_whitespace(value: Any) -> Any:
    if isinstance(value, str) and value != value.strip():
        raise ValueError("identifier cannot contain leading or trailing whitespace")
    return value


PluginId = Annotated[
    str,
    BeforeValidator(_reject_surrounding_whitespace),
    Field(
        pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)+$",
        min_length=3,
        max_length=128,
    ),
]
OpaqueId = Annotated[
    str,
    BeforeValidator(_reject_surrounding_whitespace),
    Field(min_length=1, max_length=512),
]
SourceId = Annotated[
    str,
    BeforeValidator(_reject_surrounding_whitespace),
    Field(min_length=1, max_length=256),
]
RequestId = Annotated[
    str,
    BeforeValidator(_reject_surrounding_whitespace),
    Field(min_length=1, max_length=128),
]
RevisionId = Annotated[
    str,
    BeforeValidator(_reject_surrounding_whitespace),
    Field(min_length=1, max_length=256),
]
ShortIdentifier = Annotated[
    str,
    BeforeValidator(_reject_surrounding_whitespace),
    Field(min_length=1, max_length=64),
]
HostInstanceId = Annotated[
    str,
    BeforeValidator(_reject_surrounding_whitespace),
    Field(min_length=1, max_length=128),
]


class WireModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


class SourceRef(WireModel):
    source_id: SourceId
    revision: RevisionId
    display_label: Annotated[str, Field(min_length=1, max_length=512)]
    media_type: Annotated[str, Field(min_length=1, max_length=128)]


class SourceSnapshot(WireModel):
    source: SourceRef
    content_sha256: Annotated[
        str,
        Field(
            pattern=r"^[a-f0-9]{64}$",
            description=(
                "SHA-256 of exact inline UTF-8 bytes, or of the actual raw "
                "bytes fetched for URI-backed content."
            ),
        ),
    ]
    inline_text: str | None = Field(
        default=None,
        description=(
            "Exact decoded text; no trimming, newline conversion, or Unicode "
            "normalization is permitted."
        ),
    )
    resource_uri: Annotated[str, Field(min_length=1, max_length=2048)] | None = None

    @model_validator(mode="after")
    def require_one_content_transport(self) -> SourceSnapshot:
        transports = int(self.inline_text is not None) + int(self.resource_uri is not None)
        if transports != 1:
            raise ValueError("exactly one content transport is required")
        if self.inline_text is not None:
            digest = hashlib.sha256(self.inline_text.encode("utf-8")).hexdigest()
            if digest != self.content_sha256:
                raise ValueError("content_sha256 does not match exact inline_text UTF-8 bytes")
        return self


class EvidenceLocator(WireModel):
    """Locate evidence in one exact source revision.

    ``text_span`` uses zero-based Unicode scalar-value offsets. The interval is
    left-closed and right-open: ``[start, end)``.
    """

    kind: Literal["text_span", "page", "section", "opaque"]
    start: Annotated[int, Field(ge=0)] | None = None
    end: Annotated[int, Field(gt=0)] | None = None
    page: Annotated[int, Field(ge=1)] | None = None
    section_label: Annotated[str, Field(min_length=1, max_length=512)] | None = None
    opaque_locator: Annotated[str, Field(min_length=1, max_length=1024)] | None = None

    @model_validator(mode="after")
    def validate_locator(self) -> EvidenceLocator:
        other_values = {
            "text_span": (self.page, self.section_label, self.opaque_locator),
            "page": (self.start, self.end, self.section_label, self.opaque_locator),
            "section": (self.start, self.end, self.page, self.opaque_locator),
            "opaque": (self.start, self.end, self.page, self.section_label),
        }
        if self.kind == "text_span":
            if self.start is None or self.end is None or self.end <= self.start:
                raise ValueError("text_span requires an increasing start/end pair")
        elif self.kind == "page" and self.page is None:
            raise ValueError("page locator requires page")
        elif self.kind == "section" and self.section_label is None:
            raise ValueError("section locator requires section_label")
        elif self.kind == "opaque" and self.opaque_locator is None:
            raise ValueError("opaque locator requires opaque_locator")
        if any(value is not None for value in other_values[self.kind]):
            raise ValueError(f"{self.kind} cannot carry fields from another locator kind")
        return self


class EvidenceRef(WireModel):
    evidence_id: Annotated[
        str,
        BeforeValidator(_reject_surrounding_whitespace),
        Field(min_length=1, max_length=256),
    ]
    source_ref: SourceId
    source_revision: RevisionId
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
    collection_item = items_by_key.get((selected.plugin_id, selected.opaque_id))
    if collection_item is None:
        raise ValueError("selected focus must belong to the current collection")
    if collection_item != selected:
        raise ValueError("selected focus must exactly match its collection item")


class FocusContext(WireModel):
    collection: tuple[OpaqueFocus, ...] = ()
    selected: OpaqueFocus | None = None

    @model_validator(mode="after")
    def selected_focus_must_belong_to_collection(self) -> FocusContext:
        _validate_focus_membership(self.collection, self.selected)
        return self


class DomainRequest(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: RequestId
    query: Annotated[str, Field(min_length=1, max_length=16000)]
    locale: Annotated[str, Field(min_length=2, max_length=32)] = "und"
    source_scope: tuple[SourceRef, ...]
    focus: FocusContext = FocusContext()
    deadline_ms: Annotated[int, Field(ge=50, le=300000)] = 10000

    @model_validator(mode="after")
    def validate_focus_scope(self) -> DomainRequest:
        source_ids = [source.source_id for source in self.source_scope]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_scope contains duplicate source IDs")
        allowed_source_ids = set(source_ids)
        focus_plugin_ids = {item.plugin_id for item in self.focus.collection}
        if len(focus_plugin_ids) > 1:
            raise ValueError("a focus collection cannot mix plugins")
        for item in self.focus.collection:
            if not set(item.source_refs).issubset(allowed_source_ids):
                raise ValueError("focus points outside the request source scope")
        return self


class ProbeResult(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: RequestId
    plugin_id: PluginId
    disposition: Literal["claim", "abstain"]
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence_source_refs: tuple[SourceId, ...] = ()
    reason_code: Annotated[str, Field(pattern=r"^[a-z0-9_.-]+$", max_length=128)] | None = None

    @model_validator(mode="after")
    def validate_abstention(self) -> ProbeResult:
        if self.disposition == "abstain" and (self.score != 0.0 or self.evidence_source_refs):
            raise ValueError("abstention must have zero score and no evidence references")
        return self


class FocusUpdate(WireModel):
    mode: Literal["preserve", "replace_collection", "clear"] = "preserve"
    items: tuple[OpaqueFocus, ...] = ()
    selected: OpaqueFocus | None = None

    @model_validator(mode="after")
    def validate_update(self) -> FocusUpdate:
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
    request_id: RequestId
    plugin_id: PluginId
    status: Literal["handled", "abstain", "retryable_error", "fatal_error"]
    answer_markdown: Annotated[str, Field(max_length=50000)] = ""
    focus_update: FocusUpdate = FocusUpdate()
    evidence: tuple[EvidenceRef, ...] = ()
    warnings: tuple[Annotated[str, Field(max_length=1000)], ...] = ()
    error: PluginError | None = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> DomainResult:
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
    request_id: RequestId
    upserts: tuple[SourceSnapshot, ...] = ()
    removed_source_ids: tuple[SourceId, ...] = ()


class SourceSyncResult(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: RequestId
    plugin_id: PluginId
    status: Literal["ok", "retryable_error", "fatal_error"]
    accepted_source_ids: tuple[SourceId, ...] = ()
    rejected_source_ids: tuple[SourceId, ...] = ()
    cache_revision: RevisionId | None = None
    error: PluginError | None = None

    @model_validator(mode="after")
    def validate_sync_status(self) -> SourceSyncResult:
        if self.status == "ok" and self.error is not None:
            raise ValueError("successful sync cannot carry an error")
        if self.status != "ok" and self.error is None:
            raise ValueError("failed sync requires an error payload")
        return self


class PluginDescribeRequest(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: RequestId


class PluginManifest(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: RequestId
    plugin_id: PluginId
    plugin_version: ShortIdentifier
    schema_version: ShortIdentifier
    display_name: Annotated[str, Field(min_length=1, max_length=256)]
    transport_modes: tuple[Literal["in_process", "mcp"], ...]
    permissions: tuple[
        Literal["source_content", "persistent_storage", "network", "model_access"],
        ...,
    ] = ()


class PluginStartRequest(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: RequestId
    host_instance_id: HostInstanceId
    storage_uri: Annotated[str, Field(min_length=1, max_length=2048)] | None = None
    config: dict[str, JsonValue] = Field(default_factory=dict)


class PluginStopRequest(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: RequestId
    host_instance_id: HostInstanceId


class LifecycleResult(WireModel):
    protocol_version: Literal["1.0"] = PROTOCOL_VERSION
    request_id: RequestId
    plugin_id: PluginId
    status: Literal["ok", "retryable_error", "fatal_error"]
    error: PluginError | None = None

    @model_validator(mode="after")
    def validate_lifecycle_status(self) -> LifecycleResult:
        if self.status == "ok" and self.error is not None:
            raise ValueError("successful lifecycle operation cannot carry an error")
        if self.status != "ok" and self.error is None:
            raise ValueError("failed lifecycle operation requires an error payload")
        return self
