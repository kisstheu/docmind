from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_distribution_metadata_and_dependency_boundary() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    assert project["name"] == "docmind-domain-sdk"
    assert project["version"] == "0.1.0"
    assert project["requires-python"] == ">=3.12"
    assert project["dependencies"] == ["pydantic>=2.7,<3"]


def test_typed_marker_and_schema_are_package_data() -> None:
    package_data = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["tool"]["setuptools"]["package-data"]["docmind_domain_sdk"]
    assert package_data == ["py.typed", "schemas/*.json"]
    package_root = PROJECT_ROOT / "src" / "docmind_domain_sdk"
    assert (package_root / "py.typed").is_file()
    assert (package_root / "schemas" / "protocol-1.0.schema.json").is_file()
