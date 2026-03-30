from __future__ import annotations

import re
from pathlib import Path

from ai.capability_common import format_bytes


def calc_repo_total_bytes(repo_state) -> int:
    doc_records = getattr(repo_state, "doc_records", None) or []
    if not doc_records:
        return 0

    total_bytes = 0
    for record in doc_records:
        if not isinstance(record, dict):
            continue
        total_bytes += int(record.get("file_size", 0) or record.get("size", 0) or record.get("bytes", 0) or 0)

    return total_bytes


def _answer_count(paths: list[str]) -> tuple[str, str]:
    return f"当前知识库共有 {len(paths)} 个文件。", "count"


def _answer_total_size(repo_state) -> tuple[str, str]:
    total_bytes = calc_repo_total_bytes(repo_state)
    return f"当前知识库里这些文档总共约占 {format_bytes(total_bytes)} 空间。", "total_size"


def _normalize_ref_key(text: str) -> str:
    return (text or "").strip().replace("\\", "/").lower().replace(" ", "")


def _extract_explicit_file_refs(text: str) -> list[str]:
    q = (text or "").strip()
    if not q:
        return []

    pattern = re.compile(
        r"([A-Za-z0-9_\-\u4e00-\u9fa5\s]+?\.(?:txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx))",
        flags=re.IGNORECASE,
    )
    refs: list[str] = []
    for m in pattern.findall(q):
        ref = m.strip().strip("，。！？；：,.!?;:")
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _extract_excluded_file_refs(text: str) -> list[str]:
    q = (text or "").strip()
    if not q:
        return []

    refs: list[str] = []
    pattern = re.compile(
        r"(?:除了|除开|除去)\s*([A-Za-z0-9_\-\u4e00-\u9fa5\s]+?\.(?:txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx))",
        flags=re.IGNORECASE,
    )
    for m in pattern.findall(q):
        ref = m.strip().strip("，。！？；：,.!?;:")
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def _resolve_repo_path_by_reference(source_ref: str | None, repo_paths: list[str]) -> str | None:
    if not source_ref:
        return None

    ref = source_ref.strip().replace("\\", "/")
    if not ref:
        return None

    ref_norm = _normalize_ref_key(ref)
    for pfx in ("这个", "那个", "那", "该", "这", "把"):
        if ref_norm.startswith(pfx):
            ref_norm = ref_norm[len(pfx):]

    for path in repo_paths:
        if _normalize_ref_key(path) == ref_norm:
            return path

    for path in repo_paths:
        norm_path = _normalize_ref_key(path)
        if norm_path.endswith(ref_norm) or ref_norm.endswith(norm_path):
            return path

    ref_name = _normalize_ref_key(Path(ref).name)
    for path in repo_paths:
        if _normalize_ref_key(Path(path).name) == ref_name:
            return path

    return None


def _infer_peer_paths_from_excluded(excluded_path: str, repo_paths: list[str]) -> list[str]:
    stem = Path(excluded_path).stem.lower()
    raw_tokens = re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fa5]{2,}", stem)
    noise = {"副本", "版本", "最终", "final", "copy", "draft", "test"}

    tokens: list[str] = []
    for tok in raw_tokens:
        if tok in noise or re.fullmatch(r"\d+", tok):
            continue
        if tok not in tokens:
            tokens.append(tok)

    if not tokens:
        return []

    peers: list[str] = []
    for path in repo_paths:
        norm_stem = _normalize_ref_key(Path(path).stem)
        if any(tok in norm_stem for tok in tokens):
            peers.append(path)

    return peers if len(peers) >= 2 else []


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        result.append(x)
    return result


def _resolve_size_compare_targets(question: str, last_user_question: str | None, repo_paths: list[str]) -> list[str]:
    current_refs = _extract_explicit_file_refs(question)
    last_refs = _extract_explicit_file_refs(last_user_question)

    resolved_current = _dedupe_keep_order(
        [p for p in (_resolve_repo_path_by_reference(r, repo_paths) for r in current_refs) if p]
    )
    if len(resolved_current) >= 2:
        return resolved_current

    excluded_refs = _extract_excluded_file_refs(question) + _extract_excluded_file_refs(last_user_question)
    resolved_excluded = _dedupe_keep_order(
        [p for p in (_resolve_repo_path_by_reference(r, repo_paths) for r in excluded_refs) if p]
    )

    if resolved_excluded:
        peers: list[str] = []
        for ex in resolved_excluded:
            for p in _infer_peer_paths_from_excluded(ex, repo_paths):
                if p not in peers:
                    peers.append(p)
        candidates = [p for p in peers if p not in resolved_excluded]
        if len(candidates) >= 2:
            return candidates

    resolved_last = _dedupe_keep_order(
        [p for p in (_resolve_repo_path_by_reference(r, repo_paths) for r in last_refs) if p]
    )
    if len(resolved_last) >= 2:
        return resolved_last

    return []


def _build_path_size_map(repo_state) -> dict[str, int]:
    size_map: dict[str, int] = {}

    for record in getattr(repo_state, "doc_records", []) or []:
        if not isinstance(record, dict):
            continue
        path = record.get("path")
        if not path:
            continue
        size = int(record.get("file_size", 0) or record.get("size", 0) or record.get("bytes", 0) or 0)
        if size > 0:
            size_map[path] = size

    return size_map


def _answer_size_consistency(question: str, repo_state, last_user_question: str | None = None) -> tuple[str, str]:
    paths = list(repo_state.paths)
    targets = _resolve_size_compare_targets(question, last_user_question, paths)

    if len(targets) < 2:
        return (
            "我识别到你在问文件大小是否一致，但当前缺少明确比较范围。"
            "你可以直接说：`A 和 B 大小一致吗`，或者 `除了 A，其他几个大小一致吗`。",
            "size_consistency",
        )

    size_map = _build_path_size_map(repo_state)
    rows: list[tuple[str, int]] = []
    missing: list[str] = []
    for p in targets:
        size = size_map.get(p)
        if size is None:
            missing.append(p)
        else:
            rows.append((p, size))

    if missing:
        listed = "\n".join(f"- {p}" for p in missing[:10])
        return (
            "以下文件暂时没有可用大小元数据，无法完成一致性比较：\n" + listed,
            "size_consistency",
        )

    size_values = {s for _, s in rows}
    lines = [f"- {p}：{format_bytes(s)}" for p, s in rows]
    if len(size_values) == 1:
        size_text = format_bytes(rows[0][1])
        answer = (
            f"在当前比较范围内，这 {len(rows)} 个文件大小一致（均为 {size_text}）：\n"
            + "\n".join(lines)
        )
    else:
        answer = (
            f"在当前比较范围内，这 {len(rows)} 个文件大小不一致：\n"
            + "\n".join(lines)
        )
    return answer, "size_consistency"


def _answer_format(all_files) -> tuple[str, str]:
    suffixes = sorted({file.suffix.lower() or "[无后缀]" for file in all_files})
    answer = "当前知识库中的文件格式有：\n" + "\n".join(f"- {suffix}" for suffix in suffixes)
    return answer, "format"
