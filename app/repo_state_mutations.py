from __future__ import annotations

from pathlib import Path

import numpy as np


def apply_repo_state_rename(repo_state, *, notes_dir: Path, old_rel_path: str, new_rel_path: str) -> None:
    old_rel = old_rel_path
    new_rel = new_rel_path

    repo_state.paths = [new_rel if p == old_rel else p for p in repo_state.paths]
    repo_state.chunk_paths = [new_rel if p == old_rel else p for p in repo_state.chunk_paths]

    new_doc_records = []
    for record in getattr(repo_state, "doc_records", []) or []:
        if isinstance(record, dict) and record.get("path") == old_rel:
            patched = dict(record)
            patched["path"] = new_rel
            new_doc_records.append(patched)
        else:
            new_doc_records.append(record)
    repo_state.doc_records = new_doc_records

    patched_info = []
    for line in getattr(repo_state, "file_info_list", []) or []:
        if isinstance(line, str):
            patched_info.append(line.replace(old_rel, new_rel, 1))
        else:
            patched_info.append(line)
    repo_state.file_info_list = patched_info

    if isinstance(repo_state.earliest_note, str):
        repo_state.earliest_note = repo_state.earliest_note.replace(old_rel, new_rel, 1)
    if isinstance(repo_state.latest_note, str):
        repo_state.latest_note = repo_state.latest_note.replace(old_rel, new_rel, 1)

    if getattr(repo_state, "all_files", None):
        replaced = []
        for p in repo_state.all_files:
            try:
                rel = Path(p).resolve().relative_to(notes_dir.resolve()).as_posix()
            except Exception:
                replaced.append(p)
                continue
            if rel == old_rel:
                replaced.append((notes_dir / new_rel).resolve())
            else:
                replaced.append(p)
        repo_state.all_files = replaced


def _rebuild_earliest_latest(repo_state) -> None:
    info_list = getattr(repo_state, "file_info_list", None) or []
    if not info_list:
        repo_state.earliest_note = "N/A"
        repo_state.latest_note = "N/A"
        return

    repo_state.earliest_note = info_list[0]
    repo_state.latest_note = info_list[-1]


def apply_repo_state_delete(repo_state, *, notes_dir: Path, old_rel_path: str) -> None:
    old_rel = old_rel_path
    if not getattr(repo_state, "paths", None):
        return

    doc_remove_indices = [i for i, p in enumerate(repo_state.paths) if p == old_rel]
    if not doc_remove_indices:
        return

    remove_set = set(doc_remove_indices)
    repo_state.paths = [p for i, p in enumerate(repo_state.paths) if i not in remove_set]
    repo_state.docs = [d for i, d in enumerate(repo_state.docs) if i not in remove_set]
    repo_state.file_times = [t for i, t in enumerate(repo_state.file_times) if i not in remove_set]
    repo_state.file_info_list = [x for i, x in enumerate(repo_state.file_info_list) if i not in remove_set]
    repo_state.doc_records = [r for i, r in enumerate(repo_state.doc_records) if i not in remove_set]

    if getattr(repo_state, "embeddings", None) is not None:
        try:
            repo_state.embeddings = np.delete(repo_state.embeddings, doc_remove_indices, axis=0)
        except Exception:
            pass

    chunk_remove_indices = [i for i, p in enumerate(getattr(repo_state, "chunk_paths", []) or []) if p == old_rel]
    chunk_remove_set = set(chunk_remove_indices)
    repo_state.chunk_paths = [p for i, p in enumerate(repo_state.chunk_paths) if i not in chunk_remove_set]
    repo_state.chunk_texts = [t for i, t in enumerate(repo_state.chunk_texts) if i not in chunk_remove_set]
    repo_state.chunk_meta = [m for i, m in enumerate(repo_state.chunk_meta) if i not in chunk_remove_set]
    repo_state.chunk_file_times = [t for i, t in enumerate(repo_state.chunk_file_times) if i not in chunk_remove_set]

    if getattr(repo_state, "chunk_embeddings", None) is not None and chunk_remove_indices:
        try:
            repo_state.chunk_embeddings = np.delete(repo_state.chunk_embeddings, chunk_remove_indices, axis=0)
        except Exception:
            pass

    if getattr(repo_state, "all_files", None):
        kept = []
        for p in repo_state.all_files:
            try:
                rel = Path(p).resolve().relative_to(notes_dir.resolve()).as_posix()
            except Exception:
                kept.append(p)
                continue
            if rel != old_rel:
                kept.append(p)
        repo_state.all_files = kept

    _rebuild_earliest_latest(repo_state)
