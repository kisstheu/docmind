from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = (
    PROJECT_ROOT
    / "src"
    / "docmind_domain_sdk"
    / "schemas"
    / "protocol-1.0.schema.json"
)
GENERATOR = PROJECT_ROOT / "scripts" / "generate_json_schema.py"
EXPECTED_DEFINITIONS = {
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
}


def test_schema_is_draft_2020_12_bundle_with_neutral_id() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "urn:docmind:domain-plugin-protocol:1.0"
    assert schema["x-protocol-version"] == "1.0"
    assert EXPECTED_DEFINITIONS.issubset(schema["$defs"])
    assert {
        "WireModel",
        "PluginId",
        "OpaqueId",
        "SourceId",
        "RequestId",
        "RevisionId",
        "ShortIdentifier",
        "HostInstanceId",
    }.isdisjoint(schema["$defs"])


def test_schema_generator_reports_no_drift() -> None:
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_schema_generation_is_deterministic() -> None:
    before = SCHEMA_PATH.read_bytes()
    subprocess.run([sys.executable, str(GENERATOR)], check=True, capture_output=True)
    assert SCHEMA_PATH.read_bytes() == before
