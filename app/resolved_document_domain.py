from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePath


@dataclass(frozen=True)
class ResolvedRepoDocument:
    path: str
    text: str


def resolve_repo_document(repo_state, resolved_path: str | None) -> ResolvedRepoDocument | None:
    """Map one exact RepoState path to its aligned full document, or fail closed."""
    target = str(resolved_path or "").strip()
    paths = list(getattr(repo_state, "paths", []) or [])
    docs = list(getattr(repo_state, "docs", []) or [])
    if not target or len(paths) != len(docs):
        return None

    matching_indices = [
        index
        for index, path in enumerate(paths)
        if str(path or "").strip() == target
    ]
    if len(matching_indices) != 1:
        return None

    text = docs[matching_indices[0]]
    if not isinstance(text, str) or not text.strip():
        return None
    return ResolvedRepoDocument(path=target, text=text)


def has_unique_repo_display_name(repo_state, resolved_path: str | None) -> bool:
    """Reject an inferred retrieval focus when another path has the same display stem."""
    target = str(resolved_path or "").strip()
    if not target:
        return False

    def normalized_stem(path: str) -> str:
        stem = PurePath(path.replace("\\", "/")).stem.casefold()
        return re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "", stem)

    target_stem = normalized_stem(target)
    if not target_stem:
        return False
    matches = [
        str(path or "").strip()
        for path in list(getattr(repo_state, "paths", []) or [])
        if normalized_stem(str(path or "").strip()) == target_stem
    ]
    return matches == [target]
