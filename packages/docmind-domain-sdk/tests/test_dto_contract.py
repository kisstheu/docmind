from __future__ import annotations

import hashlib
import json
import unicodedata
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

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

    assert __version__ == "0.2.0"
    assert PROTOCOL_VERSION == "1.1"


def test_request_options_default_to_an_independent_empty_object() -> None:
    first = DomainRequest(request_id="request-1", query="Question", source_scope=())
    second = DomainRequest(request_id="request-2", query="Question", source_scope=())

    assert first.options == second.options == {}
    assert first.options is not second.options


def test_request_options_round_trip_nested_json_and_valid_scalars() -> None:
    options = {
        "text": "value",
        "enabled": True,
        "missing": None,
        "count": 3,
        "ratio": 0.5,
        "nested": {"items": ["x", False, None, 7, 1.25]},
    }
    request = DomainRequest(
        request_id="request-options",
        query="Question",
        source_scope=(),
        options=options,
    )

    assert request.options == options
    assert DomainRequest.model_validate_json(request.model_dump_json()) == request
    assert json.loads(request.model_dump_json())["options"] == options


def test_request_options_accept_mapping_input_and_store_a_plain_isolated_dict() -> None:
    original = {"nested": [1, {"enabled": True}]}
    request = DomainRequest(
        request_id="request-mapping",
        query="Question",
        source_scope=(),
        options=MappingProxyType(original),
    )

    assert type(request.options) is dict
    assert request.options == original
    assert request.options is not original


@pytest.mark.parametrize(
    "invalid",
    [
        Path("private-value"),
        b"private-value",
        bytearray(b"private-value"),
        Decimal("1.5"),
        {"private-value"},
        frozenset({"private-value"}),
        ("private-value",),
        object(),
        lambda: "private-value",
    ],
)
def test_request_options_reject_non_json_python_types_without_coercion(invalid) -> None:
    with pytest.raises(ValidationError):
        DomainRequest(
            request_id="request-invalid",
            query="Question",
            source_scope=(),
            options={"value": invalid},
        )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_request_options_reject_nonfinite_numbers(invalid: float) -> None:
    with pytest.raises(ValidationError):
        DomainRequest(
            request_id="request-nonfinite",
            query="Question",
            source_scope=(),
            options={"value": invalid},
        )


def test_request_options_reject_cycles_and_non_string_keys_without_leaking_values() -> None:
    sentinel = "PRIVATE-SENTINEL-DO-NOT-ECHO"
    cyclic_dict = {sentinel: None}
    cyclic_dict[sentinel] = cyclic_dict
    cyclic_list = []
    cyclic_list.append(cyclic_list)
    invalid_options = (cyclic_dict, {"items": cyclic_list}, {1: sentinel})

    for options in invalid_options:
        with pytest.raises(ValidationError) as caught:
            DomainRequest(
                request_id="request-invalid-tree",
                query="Question",
                source_scope=(),
                options=options,
            )
        assert sentinel not in str(caught.value)


def test_request_options_are_isolated_from_all_caller_containers() -> None:
    original_item = {"label": "before"}
    original_list = [original_item]
    original = {"nested": {"items": original_list}}
    request = DomainRequest(
        request_id="request-isolation",
        query="Question",
        source_scope=(),
        options=original,
    )

    original_item["label"] = "after"
    original_list.append("after")
    original["nested"]["new"] = "after"

    assert request.options == {"nested": {"items": [{"label": "before"}]}}
    assert request.options is not original
    assert request.options["nested"] is not original["nested"]
    assert request.options["nested"]["items"] is not original_list
    assert DomainRequest.model_validate_json(request.model_dump_json()).options == request.options


def test_reused_noncyclic_container_is_accepted_and_copied_per_path() -> None:
    shared = {"items": [1, 2]}
    request = DomainRequest(
        request_id="request-shared-container",
        query="Question",
        source_scope=(),
        options={"first": shared, "second": shared},
    )

    assert request.options["first"] == request.options["second"] == shared
    assert request.options["first"] is not request.options["second"]


def test_request_options_participate_in_value_equality_without_hash_contract() -> None:
    common = {"request_id": "request-equality", "query": "Question", "source_scope": ()}
    first = DomainRequest(**common, options={"a": 1, "b": {"c": [2]}})
    reordered = DomainRequest(**common, options={"b": {"c": [2]}, "a": 1})
    different = DomainRequest(**common, options={"a": 2, "b": {"c": [2]}})

    assert first == reordered
    assert first != different
    with pytest.raises(TypeError):
        hash(first)


def test_protocol_1_0_request_requires_options_to_be_omitted_and_reserializes_without_it() -> None:
    payload = {
        "protocol_version": "1.0",
        "request_id": "legacy-request",
        "query": "Legacy question",
        "source_scope": [],
    }
    request = DomainRequest.model_validate(payload)
    json_request = DomainRequest.model_validate_json(json.dumps(payload))

    assert request.options == json_request.options == {}
    assert "options" not in request.model_dump(mode="json")
    assert "options" not in json.loads(request.model_dump_json())
    assert request.model_dump(mode="json")["query"] == payload["query"]

    for explicit_options in ({}, {"value": 1}):
        with pytest.raises(ValidationError):
            DomainRequest(**payload, options=explicit_options)
        with pytest.raises(ValidationError):
            DomainRequest.model_validate_json(
                json.dumps({**payload, "options": explicit_options})
            )


def test_request_validation_errors_hide_complete_input_values() -> None:
    sensitive_key = "secret_option_key_93841"
    sensitive_value = "SECRET_OPTION_VALUE_DO_NOT_LEAK_93841"
    invalid_options = {sensitive_key: Path(sensitive_value)}

    with pytest.raises(ValidationError) as caught:
        DomainRequest(
            request_id="request-redacted",
            query="Question",
            source_scope=(),
            options=invalid_options,
        )

    for text in (str(caught.value), repr(caught.value)):
        assert "invalid_json_type" in text
        assert sensitive_key not in text
        assert sensitive_value not in text
        assert repr(invalid_options) not in text


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
