from __future__ import annotations

from pathlib import Path


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
