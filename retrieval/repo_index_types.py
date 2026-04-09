from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List


@dataclass
class RepoState:
    docs: List[str]
    doc_records: List[dict]
    paths: List[str]
    file_times: List[datetime.datetime]
    file_info_list: List[str]
    chunk_texts: List[str]
    chunk_paths: List[str]
    chunk_meta: List[dict]
    chunk_file_times: List[datetime.datetime]
    all_files: List[Path]
    embeddings: Any
    chunk_embeddings: Any
    earliest_note: str
    latest_note: str


@dataclass
class ScanEntry:
    path: str
    file_time: datetime.datetime
    fingerprint: str
    size_kb: float


@dataclass
class ScannedRepo:
    entries: List[ScanEntry]
    paths: List[str]
    file_times: List[datetime.datetime]
    file_info_list: List[str]
    all_files: List[Path]
    earliest_note: str
    latest_note: str
    notes_dir: Path

    def to_legacy_dict(self) -> dict:
        return {
            "entries": [
                {
                    "path": e.path,
                    "file_time": e.file_time,
                    "fingerprint": e.fingerprint,
                    "size_kb": e.size_kb,
                }
                for e in self.entries
            ],
            "paths": self.paths,
            "file_times": self.file_times,
            "file_info_list": self.file_info_list,
            "all_files": self.all_files,
            "earliest_note": self.earliest_note,
            "latest_note": self.latest_note,
            "notes_dir": self.notes_dir,
        }


@dataclass
class CacheSnapshot:
    manifest: dict[str, str]
    doc_cache: dict[str, dict]
    chunk_cache: dict[str, dict]
    usable: bool


@dataclass
class ManifestDiff:
    added_paths: List[str]
    modified_paths: List[str]
    unchanged_paths: List[str]
    deleted_paths: List[str]

    @property
    def changed_paths(self) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for path in self.added_paths + self.modified_paths:
            if path not in seen:
                merged.append(path)
                seen.add(path)
        return merged


@dataclass
class ReuseResult:
    new_doc_cache: dict[str, dict]
    new_chunk_cache: dict[str, dict]
    reused_count: int
    promoted_modified_paths: List[str]


@dataclass
class FileReadResult:
    path: str
    doc: str
    file_time: datetime.datetime
    file_size: int
    file_info: str
    used_sidecar: bool


@dataclass
class PreparedFileBuild:
    path: str
    fingerprint: str
    file_record: FileReadResult
    shadow_tags: str
    scene_tags: str
    scene_tags_version: int
    chunk_texts: List[str]
    chunk_meta: List[dict]


@dataclass
class IndexBuildContext:
    notes_dir: Path
    model_emb: Any
    logger: Any
    ollama_api_url: str
    ollama_model: str
    tag_mode: str = "statistical"
    tag_concurrency: int = 1
    ollama_timeout_sec: float = 8.0
    ollama_max_retries: int = 0
