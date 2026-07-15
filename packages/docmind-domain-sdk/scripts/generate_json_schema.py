from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic.json_schema import models_json_schema


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCHEMA_PATH = (
    SRC_ROOT
    / "docmind_domain_sdk"
    / "schemas"
    / "protocol-1.0.schema.json"
)
sys.path.insert(0, str(SRC_ROOT))

from docmind_domain_sdk import (  # noqa: E402
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


DTO_MODELS = (
    SourceRef,
    SourceSnapshot,
    EvidenceLocator,
    EvidenceRef,
    OpaqueFocus,
    FocusContext,
    DomainRequest,
    ProbeResult,
    FocusUpdate,
    PluginError,
    DomainResult,
    SourceSyncRequest,
    SourceSyncResult,
    PluginDescribeRequest,
    PluginManifest,
    PluginStartRequest,
    PluginStopRequest,
    LifecycleResult,
)


def render_schema() -> str:
    _, generated = models_json_schema(
        [(model, "validation") for model in DTO_MODELS],
        title="DocMind Domain Plugin Protocol 1.0",
    )
    bundle = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:docmind:domain-plugin-protocol:1.0",
        "title": "DocMind Domain Plugin Protocol 1.0",
        "description": "Neutral wire DTO definitions for protocol 1.0.",
        "x-protocol-version": "1.0",
        "$defs": generated["$defs"],
    }
    return json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail when the committed schema differs from generated output",
    )
    args = parser.parse_args()
    rendered = render_schema()
    if args.check:
        if not SCHEMA_PATH.exists() or SCHEMA_PATH.read_text(encoding="utf-8") != rendered:
            print(f"schema drift detected: {SCHEMA_PATH}", file=sys.stderr)
            return 1
        print(f"schema is current: {SCHEMA_PATH}")
        return 0
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
