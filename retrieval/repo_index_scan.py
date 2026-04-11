from __future__ import annotations

import os
from pathlib import Path
from typing import List

from retrieval.query_utils import SUPPORTED_EXT

_EXCLUDED_PARTS = {
    ".venv",
    ".idea",
    ".git",
    ".SynologyWorkingDirectory",
    "__pycache__",
    ".docmind_trash",
}
_MAX_FILE_BYTES = 500 * 1024
_MAX_IMAGE_FILE_BYTES = 5 * 1024 * 1024
_MAX_PDF_FILE_BYTES = 100 * 1024 * 1024
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _heavy_pdf_enabled() -> bool:
    return (os.getenv("DOCMIND_ENABLE_HEAVY_PDF") or "").strip().lower() in {"1", "true", "yes", "on"}


def collect_all_files(notes_dir: Path) -> List[Path]:
    all_files: List[Path] = []
    for file in notes_dir.rglob("*"):
        if not _is_supported_file(file):
            continue
        all_files.append(file)
    all_files.sort(key=lambda x: x.stat().st_mtime)
    return all_files


def _is_supported_file(file: Path) -> bool:
    if not file.is_file():
        return False
    if any(part in file.parts for part in _EXCLUDED_PARTS):
        return False
    if file.suffix.lower() not in SUPPORTED_EXT:
        return False
    if file.name.endswith(".ocr.txt") or file.name.startswith("~$") or file.name.endswith(".converted.txt"):
        return False
    stat = file.stat()
    if stat.st_size == 0:
        return False
    suffix = file.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        size_limit = _MAX_IMAGE_FILE_BYTES
    elif suffix == ".pdf":
        size_limit = _MAX_PDF_FILE_BYTES if _heavy_pdf_enabled() else _MAX_FILE_BYTES
    else:
        size_limit = _MAX_FILE_BYTES
    if stat.st_size > size_limit:
        return False
    return True
