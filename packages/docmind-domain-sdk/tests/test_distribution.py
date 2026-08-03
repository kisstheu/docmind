from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHEEL_CONTENT_CHECKER = PROJECT_ROOT / "scripts" / "check_wheel_contents.py"


def _repository_distribution_artifacts() -> set[Path]:
    candidates = {
        PROJECT_ROOT / "build",
        PROJECT_ROOT / "dist",
        *PROJECT_ROOT.glob("*.whl"),
        *(PROJECT_ROOT / "src").glob("*.egg-info"),
    }
    return {path for path in candidates if path.exists()}


def test_distribution_metadata_and_dependency_boundary() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    assert project["name"] == "docmind-domain-sdk"
    assert project["version"] == "0.2.0"
    assert project["requires-python"] == ">=3.12"
    assert project["dependencies"] == ["pydantic>=2.7,<3"]
    assert project["optional-dependencies"]["test"] == [
        "pytest>=8,<10",
        "jsonschema>=4.18,<5",
    ]


def test_typed_marker_and_schema_are_package_data() -> None:
    package_data = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["tool"]["setuptools"]["package-data"]["docmind_domain_sdk"]
    assert package_data == ["py.typed", "schemas/*.json"]
    package_root = PROJECT_ROOT / "src" / "docmind_domain_sdk"
    assert (package_root / "py.typed").is_file()
    assert (package_root / "schemas" / "protocol-1.0.schema.json").is_file()
    assert (package_root / "schemas" / "protocol-1.1.schema.json").is_file()


def test_built_wheel_contains_runtime_modules_types_and_both_schemas(tmp_path: Path) -> None:
    repository_artifacts_before = _repository_distribution_artifacts()
    temporary_source = tmp_path / "docmind-domain-sdk-source"
    temporary_source.mkdir()
    shutil.copy2(PROJECT_ROOT / "pyproject.toml", temporary_source / "pyproject.toml")
    shutil.copy2(PROJECT_ROOT / "README.md", temporary_source / "README.md")
    shutil.copytree(
        PROJECT_ROOT / "src",
        temporary_source / "src",
        ignore=shutil.ignore_patterns(
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.egg-info",
        ),
    )
    wheel_directory = tmp_path / "wheel"
    wheel_directory.mkdir()
    environment = {
        **os.environ,
        "PIP_CACHE_DIR": str(tmp_path / "pip-cache"),
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--wheel-dir",
            str(wheel_directory),
            str(temporary_source),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert built.returncode == 0, f"stdout:\n{built.stdout}\nstderr:\n{built.stderr}"

    wheels = tuple(wheel_directory.glob("docmind_domain_sdk-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]
    assert wheel.is_relative_to(tmp_path)
    assert not wheel.is_relative_to(PROJECT_ROOT)

    checked = subprocess.run(
        [sys.executable, str(WHEEL_CONTENT_CHECKER), str(wheel)],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert checked.returncode == 0, (
        f"stdout:\n{checked.stdout}\nstderr:\n{checked.stderr}"
    )
    assert "wheel content check: ok" in checked.stdout

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert {
        "docmind_domain_sdk/__init__.py",
        "docmind_domain_sdk/_version.py",
        "docmind_domain_sdk/dto.py",
        "docmind_domain_sdk/errors.py",
        "docmind_domain_sdk/protocol.py",
        "docmind_domain_sdk/validation.py",
        "docmind_domain_sdk/py.typed",
        "docmind_domain_sdk/schemas/protocol-1.0.schema.json",
        "docmind_domain_sdk/schemas/protocol-1.1.schema.json",
    }.issubset(names)
    assert _repository_distribution_artifacts() == repository_artifacts_before
