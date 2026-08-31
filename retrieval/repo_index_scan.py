from __future__ import annotations

import os
from pathlib import Path
from typing import List

from retrieval.query_utils import SUPPORTED_EXT
from retrieval.repo_index_types import RejectedFile

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
_MAX_DEFAULT_PDF_FILE_BYTES = 2000 * 1024
_MAX_PDF_FILE_BYTES = 100 * 1024 * 1024
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _heavy_pdf_enabled() -> bool:
    return (os.getenv("DOCMIND_ENABLE_HEAVY_PDF") or "").strip().lower() in {"1", "true", "yes", "on"}


def collect_all_files(notes_dir: Path) -> List[Path]:
    all_files, _ = scan_files(notes_dir)
    return all_files


def scan_files(notes_dir: Path) -> tuple[List[Path], List[RejectedFile]]:
    all_files: List[Path] = []
    rejected_files: List[RejectedFile] = []
    for file in notes_dir.rglob("*"):
        decision, reason = _classify_file(file)
        if decision == "ignore":
            continue
        if decision == "index":
            all_files.append(file)
            continue
        rejected_files.append(
            RejectedFile(
                path=file.relative_to(notes_dir).as_posix(),
                category=decision,
                reason=reason,
            )
        )
    all_files.sort(key=lambda x: x.stat().st_mtime)
    rejected_files.sort(key=lambda item: item.path)
    return all_files, rejected_files


def _is_supported_file(file: Path) -> bool:
    decision, _ = _classify_file(file)
    return decision == "index"


def _classify_file(file: Path) -> tuple[str, str]:
    if not file.is_file():
        return "ignore", ""
    if any(part in file.parts for part in _EXCLUDED_PARTS):
        return "ignore", ""
    if file.name.endswith(".ocr.txt") or file.name.startswith("~$") or file.name.endswith(".converted.txt"):
        return "ignore", ""
    suffix = file.suffix.lower()
    if suffix not in SUPPORTED_EXT:
        display_suffix = suffix or "无扩展名"
        return "unsupported", f"暂不支持 {display_suffix} 文件"
    stat = file.stat()
    if stat.st_size == 0:
        return "ignore", ""
    if suffix in _IMAGE_EXTENSIONS:
        size_limit = _MAX_IMAGE_FILE_BYTES
    elif suffix == ".pdf":
        size_limit = _MAX_PDF_FILE_BYTES if _heavy_pdf_enabled() else _MAX_DEFAULT_PDF_FILE_BYTES
    else:
        size_limit = _MAX_FILE_BYTES
    if stat.st_size > size_limit:
        return "resource_limit", f"超过当前 {suffix} 文件大小上限（{size_limit // 1024}KB）"
    return "index", ""
