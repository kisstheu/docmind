from __future__ import annotations

import docmind_domain_sdk


EXPECTED_PUBLIC_API = {
    "SourceRef",
    "SourceSnapshot",
    "EvidenceLocator",
    "EvidenceRef",
    "OpaqueFocus",
    "FocusContext",
    "DomainRequest",
    "ProbeResult",
    "FocusUpdate",
    "PluginError",
    "DomainResult",
    "SourceSyncRequest",
    "SourceSyncResult",
    "PluginDescribeRequest",
    "PluginManifest",
    "PluginStartRequest",
    "PluginStopRequest",
    "LifecycleResult",
    "DomainPlugin",
    "PROTOCOL_VERSION",
    "ProtocolViolationError",
    "validate_manifest_boundary",
    "validate_lifecycle_boundary",
    "validate_sync_boundary",
    "validate_probe_boundary",
    "validate_result_boundary",
}


def test_top_level_all_is_exactly_the_confirmed_api() -> None:
    assert set(docmind_domain_sdk.__all__) == EXPECTED_PUBLIC_API
    assert len(docmind_domain_sdk.__all__) == len(EXPECTED_PUBLIC_API)
    for name in EXPECTED_PUBLIC_API:
        assert getattr(docmind_domain_sdk, name) is not None


def test_internal_objects_are_not_exposed_at_top_level() -> None:
    forbidden = {
        "WireModel",
        "PluginId",
        "OpaqueId",
        "SourceId",
        "RequestId",
        "DTO_MODELS",
        "render_schema",
        "fixtures",
    }
    assert forbidden.isdisjoint(docmind_domain_sdk.__dict__)
