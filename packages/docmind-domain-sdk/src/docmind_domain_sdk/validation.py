from __future__ import annotations

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
from .errors import ProtocolViolationError


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolViolationError(message)


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
