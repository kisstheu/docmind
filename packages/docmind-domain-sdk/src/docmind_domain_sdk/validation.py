from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .errors import ProtocolViolationError

if TYPE_CHECKING:
    from .dto import (
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


MAX_OPTIONS_ENCODED_BYTES = 16384
MAX_OPTIONS_DEPTH = 8
MAX_OPTIONS_TOTAL_KEYS = 128
MAX_OPTIONS_CONTAINER_ITEMS = 256


class _OptionsValidationFailure(ValueError):
    pass


@dataclass
class _OptionsStats:
    total_keys: int = 0
    container_items: int = 0


def _copy_json_value(
    value: Any,
    *,
    depth: int,
    active_container_ids: set[int],
    stats: _OptionsStats,
) -> Any:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _OptionsValidationFailure("invalid_json_type")
        return value
    if type(value) not in {dict, list}:
        raise _OptionsValidationFailure("invalid_json_type")
    if depth > MAX_OPTIONS_DEPTH:
        raise _OptionsValidationFailure(
            f"depth exceeds maximum {MAX_OPTIONS_DEPTH}"
        )

    container_id = id(value)
    if container_id in active_container_ids:
        raise _OptionsValidationFailure("invalid_json_type")
    active_container_ids.add(container_id)
    try:
        stats.container_items += len(value)
        if stats.container_items > MAX_OPTIONS_CONTAINER_ITEMS:
            raise _OptionsValidationFailure(
                f"container_items exceeds maximum {MAX_OPTIONS_CONTAINER_ITEMS}"
            )
        if type(value) is dict:
            stats.total_keys += len(value)
            if stats.total_keys > MAX_OPTIONS_TOTAL_KEYS:
                raise _OptionsValidationFailure(
                    f"total_keys exceeds maximum {MAX_OPTIONS_TOTAL_KEYS}"
                )
            copied: dict[str, Any] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise _OptionsValidationFailure("invalid_json_type")
                copied[key] = _copy_json_value(
                    item,
                    depth=depth + 1 if type(item) in {dict, list} else depth,
                    active_container_ids=active_container_ids,
                    stats=stats,
                )
            return copied
        return [
            _copy_json_value(
                item,
                depth=depth + 1 if type(item) in {dict, list} else depth,
                active_container_ids=active_container_ids,
                stats=stats,
            )
            for item in value
        ]
    finally:
        active_container_ids.remove(container_id)


def _validated_options_copy(options: Any) -> dict[str, Any]:
    if not isinstance(options, Mapping):
        raise _OptionsValidationFailure("invalid_json_type")
    if type(options) is not dict:
        try:
            options = dict(options)
        except Exception:
            raise _OptionsValidationFailure("invalid_json_type") from None
    copied = _copy_json_value(
        options,
        depth=1,
        active_container_ids=set(),
        stats=_OptionsStats(),
    )
    try:
        encoded = json.dumps(
            copied,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise _OptionsValidationFailure("invalid_json_type") from None
    if len(encoded) > MAX_OPTIONS_ENCODED_BYTES:
        raise _OptionsValidationFailure(
            f"encoded_bytes exceeds maximum {MAX_OPTIONS_ENCODED_BYTES}"
        )
    return copied


def validate_and_copy_options(options: Any) -> dict[str, Any]:
    """Validate one options tree and return a caller-isolated copy."""

    return _validated_options_copy(options)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolViolationError(message)


def validate_request_boundary(request: DomainRequest) -> None:
    if request.protocol_version == "1.0" and "options" in request.model_fields_set:
        raise ProtocolViolationError("invalid_json_type")
    try:
        _validated_options_copy(request.options)
    except _OptionsValidationFailure as error:
        raise ProtocolViolationError(str(error)) from None


def validate_result_boundary(
    request: DomainRequest,
    result: DomainResult,
    *,
    expected_plugin_id: str,
) -> None:
    _require(result.request_id == request.request_id, "response request_id mismatch")
    _require(result.plugin_id == expected_plugin_id, "response plugin_id mismatch")
    source_revisions = {source.source_id: source.revision for source in request.source_scope}
    allowed_source_ids = set(source_revisions)
    for evidence in result.evidence:
        _require(
            evidence.source_ref in allowed_source_ids,
            "evidence points outside the request source scope",
        )
        _require(
            evidence.source_revision == source_revisions[evidence.source_ref],
            "evidence revision does not match the request source revision",
        )
    for focus in result.focus_update.items:
        _require(focus.plugin_id == expected_plugin_id, "focus belongs to a different plugin")
        _require(
            set(focus.source_refs).issubset(allowed_source_ids),
            "focus points outside the request source scope",
        )


def validate_probe_boundary(
    request: DomainRequest,
    result: ProbeResult,
    *,
    expected_plugin_id: str,
) -> None:
    _require(result.request_id == request.request_id, "probe request_id mismatch")
    _require(result.plugin_id == expected_plugin_id, "probe plugin_id mismatch")
    allowed_source_ids = {source.source_id for source in request.source_scope}
    _require(
        set(result.evidence_source_refs).issubset(allowed_source_ids),
        "probe evidence points outside the request source scope",
    )


def validate_sync_boundary(
    request: SourceSyncRequest,
    result: SourceSyncResult,
    *,
    expected_plugin_id: str,
) -> None:
    _require(result.request_id == request.request_id, "sync request_id mismatch")
    _require(result.plugin_id == expected_plugin_id, "sync plugin_id mismatch")
    requested_source_ids = {
        *(snapshot.source.source_id for snapshot in request.upserts),
        *request.removed_source_ids,
    }
    accepted = set(result.accepted_source_ids)
    rejected = set(result.rejected_source_ids)
    _require(
        len(accepted) == len(result.accepted_source_ids),
        "accepted source IDs contain duplicates",
    )
    _require(
        len(rejected) == len(result.rejected_source_ids),
        "rejected source IDs contain duplicates",
    )
    _require(
        accepted.issubset(requested_source_ids),
        "accepted source ID was not present in the sync request",
    )
    _require(
        rejected.issubset(requested_source_ids),
        "rejected source ID was not present in the sync request",
    )
    _require(not accepted.intersection(rejected), "accepted and rejected source IDs overlap")


def validate_manifest_boundary(
    request: PluginDescribeRequest,
    result: PluginManifest,
    *,
    expected_plugin_id: str,
) -> None:
    _require(result.request_id == request.request_id, "describe request_id mismatch")
    _require(result.plugin_id == expected_plugin_id, "describe plugin_id mismatch")


def validate_lifecycle_boundary(
    request: PluginStartRequest | PluginStopRequest,
    result: LifecycleResult,
    *,
    expected_plugin_id: str,
) -> None:
    _require(result.request_id == request.request_id, "lifecycle request_id mismatch")
    _require(result.plugin_id == expected_plugin_id, "lifecycle plugin_id mismatch")
