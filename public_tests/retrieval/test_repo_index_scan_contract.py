from pathlib import Path

from retrieval.repo_index_scan import collect_all_files


def _relative_paths(root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in collect_all_files(root)]


def test_default_scan_keeps_small_document_and_rejects_oversized_pdf(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("DOCMIND_ENABLE_HEAVY_PDF", raising=False)
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "alpha.md").write_text("Synthetic note.", encoding="utf-8")
    (docs / "beta.pdf").write_bytes(b"x" * (512 * 1024 + 1))

    assert _relative_paths(tmp_path) == ["docs/alpha.md"]


def test_heavy_pdf_flag_allows_pdf_above_default_limit(tmp_path: Path, monkeypatch) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "beta.pdf").write_bytes(b"x" * (512 * 1024 + 1))
    monkeypatch.setenv("DOCMIND_ENABLE_HEAVY_PDF", "true")

    assert _relative_paths(tmp_path) == ["docs/beta.pdf"]


def test_scan_excludes_runtime_directory_and_large_text_document(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("DOCMIND_ENABLE_HEAVY_PDF", raising=False)
    runtime_dir = tmp_path / ".docmind_trash"
    runtime_dir.mkdir()
    (runtime_dir / "alpha.md").write_text("Synthetic runtime note.", encoding="utf-8")
    (tmp_path / "oversized.md").write_bytes(b"x" * (512 * 1024 + 1))

    assert _relative_paths(tmp_path) == []
