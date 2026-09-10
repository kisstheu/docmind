from __future__ import annotations

from pathlib import Path

import pytest

from retrieval.repo_index import scan_repository
from retrieval.repo_index_cache import classify_manifest_diff
from retrieval.repo_index_scan import collect_all_files
from retrieval.repo_index_types import CacheSnapshot


class _RecordingLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _relative_paths(root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in collect_all_files(root)]


def _sparse_file(path: Path, size_bytes: int) -> None:
    with path.open("wb") as file:
        file.truncate(size_bytes)


@pytest.mark.parametrize("ancestor", [".git", ".idea", ".venv"])
def test_explicit_source_root_is_independent_of_ancestor_name(tmp_path: Path, ancestor: str) -> None:
    root = tmp_path / ancestor / "selected_notes"
    root.mkdir(parents=True)
    (root / "source.txt").write_text("Synthetic source.", encoding="utf-8")
    assert _relative_paths(root) == ["source.txt"]
    assert _relative_paths(tmp_path) == []


@pytest.mark.parametrize("internal", [".git", ".idea", ".venv", "__pycache__", ".docmind_trash"])
def test_explicit_source_root_still_excludes_internal_directories(tmp_path: Path, internal: str) -> None:
    root = tmp_path / ".git" / "selected_notes"
    (root / internal).mkdir(parents=True)
    (root / "source.txt").write_text("Synthetic source.", encoding="utf-8")
    (root / internal / "hidden.txt").write_text("Synthetic excluded source.", encoding="utf-8")
    assert _relative_paths(root) == ["source.txt"]


def test_default_scan_discovers_normal_pdfs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DOCMIND_ENABLE_HEAVY_PDF", "0")
    docs = tmp_path / "docs"
    docs.mkdir()
    _sparse_file(docs / "small.pdf", 400 * 1024)
    _sparse_file(docs / "medium.pdf", 600 * 1024)
    _sparse_file(docs / "normal.pdf", 1_900_000)

    assert set(_relative_paths(tmp_path)) == {
        "docs/small.pdf",
        "docs/medium.pdf",
        "docs/normal.pdf",
    }


@pytest.mark.parametrize("size_bytes", [538114, 635796, 1674547, 1901025])
def test_default_scan_discovers_reproduced_pdf_sizes(
    tmp_path: Path,
    monkeypatch,
    size_bytes: int,
) -> None:
    monkeypatch.setenv("DOCMIND_ENABLE_HEAVY_PDF", "0")
    _sparse_file(tmp_path / f"document-{size_bytes}.pdf", size_bytes)

    assert _relative_paths(tmp_path) == [f"document-{size_bytes}.pdf"]


@pytest.mark.parametrize("size_bytes", [2048000, 2048001, 2097151, 2097152])
def test_default_pdf_limit_indexes_through_exactly_2_mib(
    tmp_path: Path,
    monkeypatch,
    size_bytes: int,
) -> None:
    monkeypatch.setenv("DOCMIND_ENABLE_HEAVY_PDF", "0")
    _sparse_file(tmp_path / f"document-{size_bytes}.pdf", size_bytes)

    assert _relative_paths(tmp_path) == [f"document-{size_bytes}.pdf"]


def test_unsupported_html_is_rejected_and_reported(tmp_path: Path) -> None:
    html = tmp_path / "example.html"
    html.write_text("<p>Synthetic document.</p>", encoding="utf-8")
    logger = _RecordingLogger()

    scanned = scan_repository(tmp_path, logger)

    assert scanned["paths"] == []
    assert scanned["rejected_files"] == [
        {
            "path": "example.html",
            "category": "unsupported",
            "reason": "暂不支持 .html 文件",
        }
    ]
    warning_text = "\n".join(logger.warnings)
    assert "example.html" in warning_text
    assert "不支持" in warning_text


def test_internal_sidecars_are_silently_ignored(tmp_path: Path) -> None:
    (tmp_path / "foo.pdf.ocr.txt").write_text("Synthetic OCR sidecar.", encoding="utf-8")
    (tmp_path / "foo.pdf.converted.txt").write_text("Synthetic converted sidecar.", encoding="utf-8")
    (tmp_path / "~$foo.docx").write_text("Synthetic lock file.", encoding="utf-8")
    logger = _RecordingLogger()

    scanned = scan_repository(tmp_path, logger)

    assert scanned["paths"] == []
    assert scanned["rejected_files"] == []
    assert logger.warnings == []


def test_pdf_resource_limit_is_rejected_and_reported(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DOCMIND_ENABLE_HEAVY_PDF", "0")
    _sparse_file(tmp_path / "oversized.pdf", 2 * 1024 * 1024 + 1)
    logger = _RecordingLogger()

    scanned = scan_repository(tmp_path, logger)

    assert scanned["paths"] == []
    assert scanned["rejected_files"][0]["path"] == "oversized.pdf"
    assert scanned["rejected_files"][0]["category"] == "resource_limit"
    assert scanned["rejected_files"][0]["reason"] == "超过当前 .pdf 文件大小上限（2 MiB）"
    warning_text = "\n".join(logger.warnings)
    assert "oversized.pdf" in warning_text
    assert "大小上限" in warning_text


def test_heavy_pdf_flag_retains_expensive_processing_opt_in(tmp_path: Path, monkeypatch) -> None:
    _sparse_file(tmp_path / "oversized.pdf", 2 * 1024 * 1024)
    monkeypatch.setenv("DOCMIND_ENABLE_HEAVY_PDF", "1")

    assert _relative_paths(tmp_path) == ["oversized.pdf"]


def test_heavy_pdf_limit_remains_inclusive_at_100_mib(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DOCMIND_ENABLE_HEAVY_PDF", "1")
    _sparse_file(tmp_path / "at-limit.pdf", 100 * 1024 * 1024)
    _sparse_file(tmp_path / "over-limit.pdf", 100 * 1024 * 1024 + 1)
    logger = _RecordingLogger()

    scanned = scan_repository(tmp_path, logger)

    assert scanned["paths"] == ["at-limit.pdf"]
    assert scanned["rejected_files"] == [
        {
            "path": "over-limit.pdf",
            "category": "resource_limit",
            "reason": "超过当前 .pdf 文件大小上限（100 MiB）",
        }
    ]


def test_rejected_files_are_separate_from_manifest_diff(tmp_path: Path) -> None:
    (tmp_path / "accepted.md").write_text("Synthetic note.", encoding="utf-8")
    (tmp_path / "example.html").write_text("<p>Synthetic document.</p>", encoding="utf-8")
    scanned = scan_repository(tmp_path, _RecordingLogger())
    manifest = {entry["path"]: entry["fingerprint"] for entry in scanned["entries"]}
    snapshot = CacheSnapshot(
        manifest={},
        doc_cache={},
        chunk_cache={},
        archived_manifest={},
        archived_doc_cache={},
        archived_chunk_cache={},
        usable=True,
    )

    diff = classify_manifest_diff(scanned["paths"], manifest, snapshot)

    assert diff.added_paths == ["accepted.md"]
    assert diff.modified_paths == []
    assert diff.deleted_paths == []
    assert diff.unchanged_paths == []
    assert [item["path"] for item in scanned["rejected_files"]] == ["example.html"]


def test_existing_supported_formats_remain_discoverable(tmp_path: Path) -> None:
    for suffix in (".txt", ".md", ".pdf", ".doc", ".docx"):
        (tmp_path / f"document{suffix}").write_bytes(b"synthetic")

    assert set(_relative_paths(tmp_path)) == {
        "document.txt",
        "document.md",
        "document.pdf",
        "document.doc",
        "document.docx",
    }


@pytest.mark.parametrize("directory", ["contract", "procurement", "project"])
def test_pdf_rule_generalizes_across_directories(
    tmp_path: Path,
    monkeypatch,
    directory: str,
) -> None:
    monkeypatch.setenv("DOCMIND_ENABLE_HEAVY_PDF", "0")
    nested = tmp_path / directory
    nested.mkdir()
    _sparse_file(nested / "document.pdf", 1_500_000)

    assert _relative_paths(tmp_path) == [f"{directory}/document.pdf"]


def test_large_text_and_internal_file_are_not_relaxed_by_pdf_contract(tmp_path: Path) -> None:
    _sparse_file(tmp_path / "oversized.txt", 512 * 1024 + 1)
    (tmp_path / "document.pdf.ocr.txt").write_text("Synthetic sidecar.", encoding="utf-8")
    logger = _RecordingLogger()

    scanned = scan_repository(tmp_path, logger)

    assert scanned["paths"] == []
    assert [item["path"] for item in scanned["rejected_files"]] == ["oversized.txt"]
    assert all("ocr.txt" not in warning for warning in logger.warnings)
