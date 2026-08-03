from __future__ import annotations

import inspect
import json

import pytest
from pydantic import ValidationError

from docmind_domain_sdk import (
    DomainPlugin,
    DomainRequest,
    LifecycleResult,
    MAX_OPTIONS_CONTAINER_ITEMS,
    MAX_OPTIONS_DEPTH,
    MAX_OPTIONS_ENCODED_BYTES,
    MAX_OPTIONS_TOTAL_KEYS,
    PluginStartRequest,
    ProbeResult,
    ProtocolViolationError,
    validate_lifecycle_boundary,
    validate_manifest_boundary,
    validate_probe_boundary,
    validate_request_boundary,
    validate_result_boundary,
    validate_sync_boundary,
)

from fixtures import (
    PLUGIN_ID,
    domain_request,
    domain_result,
    manifest_pair,
    sync_pair,
)


class AsyncShapePlugin:
    async def describe(self, request):  # noqa: ANN001
        raise NotImplementedError

    async def start(self, request):  # noqa: ANN001
        raise NotImplementedError

    async def sync_sources(self, request):  # noqa: ANN001
        raise NotImplementedError

    async def probe(self, request):  # noqa: ANN001
        raise NotImplementedError

    async def execute(self, request):  # noqa: ANN001
        raise NotImplementedError

    async def stop(self, request):  # noqa: ANN001
        raise NotImplementedError


def test_domain_plugin_is_runtime_checkable_async_shape() -> None:
    assert isinstance(AsyncShapePlugin(), DomainPlugin)
    for method_name in ("describe", "start", "sync_sources", "probe", "execute", "stop"):
        assert inspect.iscoroutinefunction(getattr(DomainPlugin, method_name))


def test_protocol_violation_error_has_the_single_confirmed_base() -> None:
    assert issubclass(ProtocolViolationError, ValueError)
    assert ProtocolViolationError.__bases__ == (ValueError,)


def _request_with_options(options) -> DomainRequest:
    return DomainRequest(
        request_id="request-options-boundary",
        query="Question",
        source_scope=(),
        options=options,
    )


def _nested_options(depth: int):
    value = "leaf"
    for _ in range(depth):
        value = {"next": value}
    return value


QUOTA_CASES = (
    pytest.param(
        {"value": "x" * (MAX_OPTIONS_ENCODED_BYTES - len('{"value":""}'))},
        {"value": "x" * (MAX_OPTIONS_ENCODED_BYTES - len('{"value":""}') + 1)},
        "encoded_bytes",
        id="encoded-bytes",
    ),
    pytest.param(
        _nested_options(MAX_OPTIONS_DEPTH),
        _nested_options(MAX_OPTIONS_DEPTH + 1),
        "depth",
        id="depth",
    ),
    pytest.param(
        {f"key-{index}": index for index in range(MAX_OPTIONS_TOTAL_KEYS)},
        {f"key-{index}": index for index in range(MAX_OPTIONS_TOTAL_KEYS + 1)},
        "total_keys",
        id="total-keys",
    ),
    pytest.param(
        {"items": [None] * (MAX_OPTIONS_CONTAINER_ITEMS - 1)},
        {"items": [None] * MAX_OPTIONS_CONTAINER_ITEMS},
        "container_items",
        id="container-items",
    ),
)


def _canonical_encoded_bytes(options) -> int:
    return len(
        json.dumps(
            options,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _assert_quota_fixture_is_exact(at_limit, over_limit, category: str) -> None:
    if category == "encoded_bytes":
        assert _canonical_encoded_bytes(at_limit) == MAX_OPTIONS_ENCODED_BYTES
        assert _canonical_encoded_bytes(over_limit) == MAX_OPTIONS_ENCODED_BYTES + 1
    elif category == "depth":
        for options, expected_depth in (
            (at_limit, MAX_OPTIONS_DEPTH),
            (over_limit, MAX_OPTIONS_DEPTH + 1),
        ):
            observed_depth = 0
            current = options
            while isinstance(current, dict):
                observed_depth += 1
                assert tuple(current) == ("next",)
                current = current["next"]
            assert current == "leaf"
            assert observed_depth == expected_depth
    elif category == "total_keys":
        assert len(at_limit) == MAX_OPTIONS_TOTAL_KEYS
        assert len(over_limit) == MAX_OPTIONS_TOTAL_KEYS + 1
    else:
        assert len(at_limit) + len(at_limit["items"]) == MAX_OPTIONS_CONTAINER_ITEMS
        assert len(over_limit) + len(over_limit["items"]) == (
            MAX_OPTIONS_CONTAINER_ITEMS + 1
        )


@pytest.mark.parametrize(
    ("at_limit", "over_limit", "message"),
    QUOTA_CASES,
)
def test_domain_request_constructor_enforces_each_options_quota(
    at_limit,
    over_limit,
    message: str,
) -> None:
    _assert_quota_fixture_is_exact(at_limit, over_limit, message)
    request = _request_with_options(at_limit)
    assert request.options == at_limit

    with pytest.raises(ValidationError) as caught:
        _request_with_options(over_limit)

    assert message in str(caught.value)
    assert message in repr(caught.value)


@pytest.mark.parametrize(
    ("at_limit", "over_limit", "message"),
    QUOTA_CASES,
)
def test_request_boundary_revalidates_each_options_quota_after_dto_bypass(
    at_limit,
    over_limit,
    message: str,
) -> None:
    request = _request_with_options(at_limit)
    validate_request_boundary(request)

    with pytest.raises(ProtocolViolationError, match=message):
        validate_request_boundary(request.model_copy(update={"options": over_limit}))


def test_request_boundary_rejects_mutated_invalid_json_without_sensitive_details() -> None:
    sensitive_key = "secret_option_key_93841"
    sensitive_value = "SECRET_OPTION_VALUE_DO_NOT_LEAK_93841"
    request = domain_request().model_copy(
        update={"options": {sensitive_key: {"value": sensitive_value.encode()}}}
    )

    with pytest.raises(ProtocolViolationError) as caught:
        validate_request_boundary(request)

    for text in (str(caught.value), repr(caught.value)):
        assert "invalid_json_type" in text
        assert sensitive_key not in text
        assert sensitive_value not in text
        assert repr(request.options) not in text


def test_request_boundary_rejects_forged_legacy_options_field() -> None:
    legacy = DomainRequest(
        protocol_version="1.0",
        request_id="legacy-request",
        query="Question",
        source_scope=(),
    )
    forged = legacy.model_copy(update={"options": {}})

    with pytest.raises(ProtocolViolationError, match="invalid_json_type"):
        validate_request_boundary(forged)


def test_all_boundary_validators_accept_matching_messages() -> None:
    manifest_request, manifest = manifest_pair()
    validate_manifest_boundary(manifest_request, manifest, expected_plugin_id=PLUGIN_ID)
    request = PluginStartRequest(request_id="start-1", host_instance_id="host-1")
    lifecycle = LifecycleResult(
        request_id=request.request_id,
        plugin_id=PLUGIN_ID,
        status="ok",
    )
    validate_lifecycle_boundary(request, lifecycle, expected_plugin_id=PLUGIN_ID)
    sync_request, sync_result = sync_pair()
    validate_sync_boundary(sync_request, sync_result, expected_plugin_id=PLUGIN_ID)
    domain = domain_request()
    probe = ProbeResult(
        request_id=domain.request_id,
        plugin_id=PLUGIN_ID,
        disposition="claim",
        score=0.5,
        evidence_source_refs=("source-1",),
    )
    validate_probe_boundary(domain, probe, expected_plugin_id=PLUGIN_ID)
    validate_result_boundary(domain, domain_result(), expected_plugin_id=PLUGIN_ID)


@pytest.mark.parametrize("validator_name", ["manifest", "lifecycle", "sync", "probe", "result"])
def test_request_id_mismatches_use_one_protocol_error(validator_name: str) -> None:
    manifest_request, manifest = manifest_pair()
    lifecycle_request = PluginStartRequest(request_id="start-1", host_instance_id="host-1")
    lifecycle = LifecycleResult(request_id="other", plugin_id=PLUGIN_ID, status="ok")
    sync_request, sync_result = sync_pair()
    domain = domain_request()
    probe = ProbeResult(
        request_id="other",
        plugin_id=PLUGIN_ID,
        disposition="abstain",
        score=0,
    )
    cases = {
        "manifest": (
            validate_manifest_boundary,
            manifest_request,
            manifest.model_copy(update={"request_id": "other"}),
        ),
        "lifecycle": (validate_lifecycle_boundary, lifecycle_request, lifecycle),
        "sync": (
            validate_sync_boundary,
            sync_request,
            sync_result.model_copy(update={"request_id": "other"}),
        ),
        "probe": (validate_probe_boundary, domain, probe),
        "result": (
            validate_result_boundary,
            domain,
            domain_result().model_copy(update={"request_id": "other"}),
        ),
    }
    validator, request, result = cases[validator_name]
    with pytest.raises(ProtocolViolationError):
        validator(request, result, expected_plugin_id=PLUGIN_ID)


def test_result_rejects_out_of_scope_and_stale_evidence() -> None:
    request = domain_request()
    result = domain_result()
    evidence = result.evidence[0]
    with pytest.raises(ProtocolViolationError, match="outside"):
        validate_result_boundary(
            request,
            result.model_copy(
                update={
                    "evidence": (
                        evidence.model_copy(update={"source_ref": "source-outside"}),
                    )
                }
            ),
            expected_plugin_id=PLUGIN_ID,
        )
    with pytest.raises(ProtocolViolationError, match="revision"):
        validate_result_boundary(
            request,
            result.model_copy(
                update={
                    "evidence": (
                        evidence.model_copy(update={"source_revision": "stale"}),
                    )
                }
            ),
            expected_plugin_id=PLUGIN_ID,
        )


def test_result_rejects_focus_owned_by_another_plugin_or_source_scope() -> None:
    request = domain_request()
    result = domain_result()
    focus = result.focus_update.items[0]
    invalid_focuses = (
        focus.model_copy(update={"plugin_id": "org.example.other"}),
        focus.model_copy(update={"source_refs": ("source-outside",)}),
    )
    for invalid_focus in invalid_focuses:
        invalid_result = result.model_copy(
            update={
                "focus_update": result.focus_update.model_copy(
                    update={"items": (invalid_focus,), "selected": invalid_focus}
                )
            }
        )
        with pytest.raises(ProtocolViolationError):
            validate_result_boundary(request, invalid_result, expected_plugin_id=PLUGIN_ID)


def test_probe_rejects_out_of_scope_source() -> None:
    request = domain_request()
    result = ProbeResult(
        request_id=request.request_id,
        plugin_id=PLUGIN_ID,
        disposition="claim",
        score=0.5,
        evidence_source_refs=("source-outside",),
    )
    with pytest.raises(ProtocolViolationError, match="outside"):
        validate_probe_boundary(request, result, expected_plugin_id=PLUGIN_ID)


def test_sync_rejects_unknown_duplicate_and_overlapping_ids() -> None:
    request, result = sync_pair()
    invalid_results = (
        result.model_copy(update={"accepted_source_ids": ("unknown",)}),
        result.model_copy(update={"accepted_source_ids": ("source-1", "source-1")}),
        result.model_copy(
            update={
                "accepted_source_ids": ("source-1",),
                "rejected_source_ids": ("source-1",),
            }
        ),
    )
    for invalid in invalid_results:
        with pytest.raises(ProtocolViolationError):
            validate_sync_boundary(request, invalid, expected_plugin_id=PLUGIN_ID)


def test_plugin_id_mismatch_is_a_protocol_violation() -> None:
    request, result = manifest_pair()
    with pytest.raises(ProtocolViolationError, match="plugin_id"):
        validate_manifest_boundary(request, result, expected_plugin_id="org.example.other")
