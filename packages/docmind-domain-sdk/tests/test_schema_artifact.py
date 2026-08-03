from __future__ import annotations

import hashlib
import json
import runpy
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from docmind_domain_sdk import DomainRequest
from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_1_0_PATH = (
    PROJECT_ROOT
    / "src"
    / "docmind_domain_sdk"
    / "schemas"
    / "protocol-1.0.schema.json"
)
SCHEMA_1_1_PATH = SCHEMA_1_0_PATH.with_name("protocol-1.1.schema.json")
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


def _load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _domain_request_validator(schema: dict[str, Any]) -> Draft202012Validator:
    return Draft202012Validator({**schema, "$ref": "#/$defs/DomainRequest"})


def _assert_schema_accepts(schema: dict[str, Any], payload: object) -> None:
    _domain_request_validator(schema).validate(payload)


def _assert_schema_rejects(schema: dict[str, Any], payload: object) -> None:
    errors = list(_domain_request_validator(schema).iter_errors(payload))
    assert errors


def _domain_request_payload(protocol_version: str) -> dict[str, object]:
    focus_item = {
        "plugin_id": "org.example.synthetic",
        "opaque_id": "item-1",
        "display_label": "Synthetic item",
        "source_refs": ["source-1"],
    }
    return {
        "protocol_version": protocol_version,
        "request_id": "schema-request",
        "query": "Schema validation question",
        "locale": "en",
        "source_scope": [
            {
                "source_id": "source-1",
                "revision": "revision-1",
                "display_label": "Synthetic source",
                "media_type": "text/plain",
            }
        ],
        "focus": {"collection": [focus_item], "selected": dict(focus_item)},
        "deadline_ms": 10000,
    }


def test_schema_is_draft_2020_12_bundle_with_neutral_id() -> None:
    schema = _load_schema(SCHEMA_1_1_PATH)
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"] == "urn:docmind:domain-plugin-protocol:1.1"
    assert schema["x-protocol-version"] == "1.1"
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


def test_schema_artifacts_are_valid_draft_2020_12_schemas() -> None:
    Draft202012Validator.check_schema(_load_schema(SCHEMA_1_0_PATH))
    Draft202012Validator.check_schema(_load_schema(SCHEMA_1_1_PATH))


def test_schema_defines_recursive_nonempty_json_value() -> None:
    json_value = _load_schema(SCHEMA_1_1_PATH)["$defs"]["JsonValue"]
    assert json_value != {}
    choices = json_value["anyOf"]
    assert {choice.get("type") for choice in choices} == {
        "object",
        "array",
        "string",
        "boolean",
        "null",
        "integer",
        "number",
    }
    object_schema = next(choice for choice in choices if choice.get("type") == "object")
    array_schema = next(choice for choice in choices if choice.get("type") == "array")
    assert object_schema["additionalProperties"] == {"$ref": "#/$defs/JsonValue"}
    assert array_schema["items"] == {"$ref": "#/$defs/JsonValue"}


def test_domain_request_schema_defaults_options_and_freezes_version_branches() -> None:
    request_schema = _load_schema(SCHEMA_1_1_PATH)["$defs"]["DomainRequest"]
    options = request_schema["properties"]["options"]
    assert options == {
        "additionalProperties": {"$ref": "#/$defs/JsonValue"},
        "default": {},
        "title": "Options",
        "type": "object",
    }
    assert request_schema["oneOf"] == [
        {
            "not": {"required": ["options"]},
            "properties": {"protocol_version": {"const": "1.0"}},
            "required": ["protocol_version"],
        },
        {"properties": {"protocol_version": {"const": "1.1"}}},
    ]


def test_protocol_1_1_schema_validates_real_domain_request_version_branches() -> None:
    schema_1_0 = _load_schema(SCHEMA_1_0_PATH)
    schema_1_1 = _load_schema(SCHEMA_1_1_PATH)
    legacy = _domain_request_payload("1.0")
    current = _domain_request_payload("1.1")

    _assert_schema_accepts(schema_1_1, legacy)
    _assert_schema_rejects(schema_1_1, {**legacy, "options": {}})
    _assert_schema_rejects(schema_1_1, {**legacy, "options": {"value": 1}})

    _assert_schema_accepts(schema_1_1, current)
    _assert_schema_accepts(schema_1_1, {**current, "options": {}})
    recursive_options = {
        **current,
        "options": {
            "nested": {"items": ["value", 3, 0.5, True, None, {"leaf": "x"}]}
        },
    }
    _assert_schema_accepts(schema_1_1, recursive_options)
    _assert_schema_rejects(schema_1_1, {**current, "options": []})
    _assert_schema_rejects(schema_1_1, {**current, "options": {"value": b"not-json"}})
    _assert_schema_rejects(schema_1_1, {**current, "unknown_request_field": True})

    _assert_schema_accepts(schema_1_0, legacy)
    _assert_schema_rejects(schema_1_0, recursive_options)


def test_legacy_reserialization_matches_frozen_1_0_request_shape() -> None:
    input_payload = _domain_request_payload("1.0")
    request = DomainRequest.model_validate(input_payload)
    schema_1_0 = _load_schema(SCHEMA_1_0_PATH)
    schema_1_1 = _load_schema(SCHEMA_1_1_PATH)
    dumped_payloads: tuple[Mapping[str, object], ...] = (
        request.model_dump(mode="json"),
        json.loads(request.model_dump_json()),
    )

    assert request.protocol_version == "1.0"
    assert request.options == {}
    for payload in dumped_payloads:
        assert "options" not in payload
        assert payload["query"] == input_payload["query"]
        assert payload["request_id"] == input_payload["request_id"]
        assert payload["source_scope"] == input_payload["source_scope"]
        assert payload["focus"] == input_payload["focus"]
        assert payload["locale"] == input_payload["locale"]
        assert payload["deadline_ms"] == input_payload["deadline_ms"]
        _assert_schema_accepts(schema_1_0, payload)
        _assert_schema_accepts(schema_1_1, payload)


def test_protocol_1_0_artifact_is_byte_for_byte_immutable() -> None:
    assert hashlib.sha256(SCHEMA_1_0_PATH.read_bytes()).hexdigest() == (
        "ef119f934d9a30794fa8aa429489e213fa4183abb1ee0e34f4ddd75550a52dc3"
    )


def test_schema_generator_reports_no_drift() -> None:
    completed = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_schema_generation_is_deterministic() -> None:
    before_1_0 = SCHEMA_1_0_PATH.read_bytes()
    before_1_1 = SCHEMA_1_1_PATH.read_bytes()
    render_schema = runpy.run_path(str(GENERATOR))["render_schema"]
    first_render = render_schema()
    second_render = render_schema()

    assert first_render == second_render
    assert first_render.encode() == before_1_1
    assert SCHEMA_1_0_PATH.read_bytes() == before_1_0
    assert SCHEMA_1_1_PATH.read_bytes() == before_1_1
