from __future__ import annotations

import hashlib
import json
import unicodedata

import pytest
from pydantic import ValidationError

from docmind_domain_sdk import (
    DomainRequest,
    DomainResult,
    EvidenceLocator,
    EvidenceRef,
    FocusContext,
    FocusUpdate,
    LifecycleResult,
    OpaqueFocus,
    PluginDescribeRequest,
    PluginError,
    PluginManifest,
    PluginStartRequest,
    PluginStopRequest,
    ProbeResult,
    SourceRef,
    SourceSnapshot,
    SourceSyncRequest,
    SourceSyncResult,
)
from docmind_domain_sdk._version import __version__

from fixtures import INLINE_TEXT, PLUGIN_ID, domain_request, domain_result, source_ref, source_snapshot


def test_package_and_protocol_versions_are_independent() -> None:
    from docmind_domain_sdk import PROTOCOL_VERSION

    assert __version__ == "0.1.0"
    assert PROTOCOL_VERSION == "1.0"


def test_models_round_trip_as_json_and_forbid_extra_fields() -> None:
    request = domain_request()
    payload = request.model_dump(mode="json")
    json.dumps(payload, allow_nan=False)
    assert DomainRequest.model_validate_json(request.model_dump_json()) == request
    payload["internal_object"] = {"not": "wire data"}
    with pytest.raises(ValidationError):
        DomainRequest.model_validate(payload)


def test_every_public_dto_round_trips_through_json() -> None:
    source = source_ref()
    snapshot = source_snapshot()
    focus = OpaqueFocus(
        plugin_id=PLUGIN_ID,
        opaque_id="item-1",
        display_label="Synthetic item",
        source_refs=(source.source_id,),
    )
    error = PluginError(code="fixture.error", message="Synthetic error", retryable=True)
    models = (
        source,
        snapshot,
        EvidenceLocator(kind="page", page=1),
        EvidenceRef(
            evidence_id="evidence-1",
            source_ref=source.source_id,
            source_revision=source.revision,
            locator=EvidenceLocator(kind="section", section_label="Section A"),
        ),
        focus,
        FocusContext(collection=(focus,), selected=focus),
        domain_request(),
        ProbeResult(
            request_id="request-1",
            plugin_id=PLUGIN_ID,
            disposition="abstain",
            score=0,
        ),
        FocusUpdate(mode="replace_collection", items=(focus,), selected=focus),
        error,
        domain_result(),
        SourceSyncRequest(request_id="sync-1", upserts=(snapshot,)),
        SourceSyncResult(
            request_id="sync-1",
            plugin_id=PLUGIN_ID,
            status="ok",
            accepted_source_ids=(source.source_id,),
        ),
        PluginDescribeRequest(request_id="describe-1"),
        PluginManifest(
            request_id="describe-1",
            plugin_id=PLUGIN_ID,
            plugin_version="0.1.0",
            schema_version="1",
            display_name="Neutral fixture",
            transport_modes=("in_process",),
        ),
        PluginStartRequest(request_id="start-1", host_instance_id="host-1"),
        PluginStopRequest(request_id="stop-1", host_instance_id="host-1"),
        LifecycleResult(request_id="start-1", plugin_id=PLUGIN_ID, status="ok"),
    )
    assert len(models) == 18
    for model in models:
        payload = model.model_dump(mode="json")
        json.dumps(payload, allow_nan=False)
        assert type(model).model_validate_json(model.model_dump_json()) == model


def test_inline_text_is_preserved_exactly() -> None:
    snapshot = source_snapshot()
    assert snapshot.inline_text == INLINE_TEXT
    assert "\r\n" in snapshot.inline_text
    assert snapshot.inline_text != snapshot.inline_text.strip()
    assert snapshot.inline_text != unicodedata.normalize("NFC", snapshot.inline_text)
    assert snapshot.content_sha256 == hashlib.sha256(INLINE_TEXT.encode("utf-8")).hexdigest()
    assert SourceSnapshot.model_validate_json(snapshot.model_dump_json()).inline_text == INLINE_TEXT


def test_inline_text_hash_uses_exact_utf8_bytes() -> None:
    with pytest.raises(ValidationError, match="exact inline_text UTF-8 bytes"):
        SourceSnapshot(
            source=source_ref(),
            content_sha256=hashlib.sha256(INLINE_TEXT.strip().encode("utf-8")).hexdigest(),
            inline_text=INLINE_TEXT,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_id", " source-1"),
        ("source_id", "source-1 "),
        ("revision", "\trevision-1"),
        ("revision", "revision-1\n"),
    ],
)
def test_source_identifiers_reject_surrounding_whitespace(field: str, value: str) -> None:
    payload = source_ref().model_dump()
    payload[field] = value
    with pytest.raises(ValidationError, match="leading or trailing whitespace"):
        SourceRef.model_validate(payload)


def test_request_and_host_identifiers_are_not_trimmed() -> None:
    with pytest.raises(ValidationError, match="leading or trailing whitespace"):
        DomainRequest(
            request_id=" request-1",
            query="Question",
            source_scope=(source_ref(),),
        )
    with pytest.raises(ValidationError, match="leading or trailing whitespace"):
        PluginStartRequest(request_id="start-1", host_instance_id="host-1 ")


def test_manifest_version_identifiers_are_not_trimmed() -> None:
    with pytest.raises(ValidationError, match="leading or trailing whitespace"):
        PluginManifest(
            request_id="describe-1",
            plugin_id=PLUGIN_ID,
            plugin_version="0.1.0 ",
            schema_version="1",
            display_name="Neutral fixture",
            transport_modes=("in_process",),
        )


def test_uri_has_no_scheme_allowlist_and_hash_is_raw_byte_digest() -> None:
    raw_bytes = b"\x00synthetic\r\nbytes"
    snapshot = SourceSnapshot(
        source=source_ref(),
        content_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        resource_uri="neutral+transport:opaque-value",
    )
    assert snapshot.resource_uri == "neutral+transport:opaque-value"
    assert snapshot.content_sha256 == hashlib.sha256(raw_bytes).hexdigest()


def test_uri_must_be_nonempty_and_within_length_limit() -> None:
    digest = hashlib.sha256(b"content").hexdigest()
    with pytest.raises(ValidationError):
        SourceSnapshot(source=source_ref(), content_sha256=digest, resource_uri="")
    with pytest.raises(ValidationError):
        SourceSnapshot(source=source_ref(), content_sha256=digest, resource_uri="x" * 2049)


def test_plugin_config_accepts_json_serializable_values() -> None:
    request = PluginStartRequest(
        request_id="start-1",
        host_instance_id="host-1",
        config={"nested": [1, True, None, {"label": " value "}]},
    )
    assert json.loads(request.model_dump_json())["config"]["nested"][3]["label"] == " value "


def test_selected_focus_must_exactly_match_collection_item() -> None:
    item = OpaqueFocus(
        plugin_id=PLUGIN_ID,
        opaque_id="item-1",
        display_label="Original label",
        source_refs=("source-1",),
    )
    altered = item.model_copy(update={"display_label": "Altered label"})
    with pytest.raises(ValidationError, match="exactly match"):
        FocusContext(collection=(item,), selected=altered)
    with pytest.raises(ValidationError, match="exactly match"):
        FocusUpdate(mode="replace_collection", items=(item,), selected=altered)


def test_request_focus_must_stay_in_source_scope() -> None:
    focus = OpaqueFocus(
        plugin_id=PLUGIN_ID,
        opaque_id="item-1",
        display_label="Synthetic item",
        source_refs=("outside-source",),
    )
    with pytest.raises(ValidationError, match="outside"):
        DomainRequest(
            request_id="request-1",
            query="Question",
            source_scope=(source_ref(),),
            focus=FocusContext(collection=(focus,), selected=focus),
        )


def test_result_status_forbids_partial_nonhandled_output() -> None:
    with pytest.raises(ValidationError, match="partial output"):
        DomainResult(
            request_id="request-1",
            plugin_id=PLUGIN_ID,
            status="abstain",
            answer_markdown="Partial answer",
        )
    with pytest.raises(ValidationError, match="requires an error"):
        DomainResult(
            request_id="request-1",
            plugin_id=PLUGIN_ID,
            status="retryable_error",
        )


def test_nonfinite_numbers_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ProbeResult(
            request_id="request-1",
            plugin_id=PLUGIN_ID,
            disposition="claim",
            score=float("nan"),
        )


def test_source_snapshot_requires_exactly_one_transport() -> None:
    digest = hashlib.sha256(b"").hexdigest()
    with pytest.raises(ValidationError, match="exactly one content transport"):
        SourceSnapshot(source=source_ref(), content_sha256=digest)
    with pytest.raises(ValidationError, match="exactly one content transport"):
        SourceSnapshot(
            source=source_ref(),
            content_sha256=digest,
            inline_text="",
            resource_uri="memory:item",
        )
