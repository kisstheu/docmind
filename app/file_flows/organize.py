from __future__ import annotations

import datetime
from pathlib import Path
import re
from typing import Sequence


DEFAULT_ORGANIZE_ROOT = "_docmind_organized"
_INVALID_FOLDER_CHARS = r'<>:"/\\|?*'
_ORGANIZE_ACTION_TERMS = ("整理", "归类", "归档", "分类整理", "分文件夹")
_ORGANIZE_SCOPE_TERMS = (
    "文件",
    "文档",
    "图片",
    "截图",
    "资料",
    "结果集",
    "知识库",
    "分类",
    "板块",
    "类别",
    "目前",
    "当前",
    "刚才",
    "这些",
)
_ALL_SCOPE_TERMS = ("全部", "所有文件", "所有图片", "所有文档", "整个知识库", "全库")
_REMAINING_SCOPE_TERMS = ("其他", "其它", "其余", "剩余", "剩下", "别的")
_NON_FILE_ORGANIZE_TERMS = ("整理思路", "整理一下思路", "整理下思路", "整理需求", "整理一下需求")
_BRIEF_ORGANIZE_TERMS = ("整理一下", "整理下", "整理吧", "归类一下", "归类下", "归档一下", "归档下")


def _normalize_intent_text(text: str) -> str:
    q = (text or "").strip().lower()
    if not q:
        return ""
    q = q.replace(" ", "")
    q = q.replace("`", "").replace('"', "").replace("'", "")
    return q


def is_organize_request(question: str) -> bool:
    q = _normalize_intent_text(question)
    if not q:
        return False

    if any(term in q for term in _NON_FILE_ORGANIZE_TERMS):
        return False

    if not any(term in q for term in _ORGANIZE_ACTION_TERMS):
        return False

    has_scope_signal = any(term in q for term in _ORGANIZE_SCOPE_TERMS)
    has_category_signal = any(term in q for term in ("分类", "板块", "类别", "归类"))
    return has_scope_signal and has_category_signal


def is_brief_organize_followup_request(question: str) -> bool:
    q = _normalize_intent_text(question)
    if not q:
        return False
    if any(term in q for term in _NON_FILE_ORGANIZE_TERMS):
        return False
    return any(term in q for term in _BRIEF_ORGANIZE_TERMS)


def should_organize_all_files(question: str) -> bool:
    q = _normalize_intent_text(question)
    if not q:
        return False
    return any(term in q for term in _ALL_SCOPE_TERMS)


def should_organize_remaining_files(question: str) -> bool:
    q = _normalize_intent_text(question)
    if not q:
        return False
    return any(term in q for term in _REMAINING_SCOPE_TERMS)


def _is_already_organized_path(path: str, organize_root_name: str = DEFAULT_ORGANIZE_ROOT) -> bool:
    normalized = str(path or "").strip().replace("\\", "/").strip("/")
    if not normalized:
        return False
    return normalized == organize_root_name or normalized.startswith(f"{organize_root_name}/")


def resolve_organize_source_paths(
    *,
    question: str,
    last_result_set_items: Sequence[str] | None,
    repo_paths: Sequence[str],
) -> tuple[list[str], str]:
    normalized_repo_paths = [str(path or "").strip() for path in repo_paths if str(path or "").strip()]
    repo_path_set = set(normalized_repo_paths)
    is_remaining_scope = should_organize_remaining_files(question)
    is_all_scope = should_organize_all_files(question)

    result_paths: list[str] = []
    for item in list(last_result_set_items or []):
        clean_item = str(item or "").strip()
        if (
            clean_item
            and clean_item in repo_path_set
            and not _is_already_organized_path(clean_item)
            and clean_item not in result_paths
        ):
            result_paths.append(clean_item)

    if is_remaining_scope:
        excluded = set(result_paths)
        remaining_paths: list[str] = []
        for path in normalized_repo_paths:
            if path in excluded or _is_already_organized_path(path):
                continue
            if path not in remaining_paths:
                remaining_paths.append(path)
        return remaining_paths, "remaining"

    if not is_all_scope and result_paths:
        return result_paths, "result_set"

    deduped: list[str] = []
    seen: set[str] = set()
    for path in normalized_repo_paths:
        if _is_already_organized_path(path):
            continue
        if path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped, "all"


def sanitize_category_folder_name(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return "未分类"
    text = re.sub(rf"[{re.escape(_INVALID_FOLDER_CHARS)}]+", "_", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text[:60] or "未分类"


def _build_session_root_rel_path(organize_root_name: str = DEFAULT_ORGANIZE_ROOT) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return (Path(organize_root_name) / stamp).as_posix()


def _build_unique_target_rel_path(
    *,
    root_rel_path: str,
    category_label: str,
    source_rel_path: str,
    planned_targets: set[str],
) -> str:
    category_dir = Path(root_rel_path) / sanitize_category_folder_name(category_label)
    source_name = Path(source_rel_path).name
    source_stem = Path(source_name).stem
    source_ext = Path(source_name).suffix

    counter = 1
    while True:
        suffix = "" if counter == 1 else f"__{counter}"
        candidate = (category_dir / f"{source_stem}{suffix}{source_ext}").as_posix()
        candidate_key = candidate.lower()
        if candidate_key not in planned_targets:
            planned_targets.add(candidate_key)
            return candidate
        counter += 1


def build_category_organize_preview(
    *,
    notes_dir: Path,
    source_paths: Sequence[str],
    path_category_map: dict[str, str],
    scope: str,
    organize_root_name: str = DEFAULT_ORGANIZE_ROOT,
) -> tuple[str, dict] | tuple[None, None]:
    session_root = _build_session_root_rel_path(organize_root_name=organize_root_name)
    planned_targets: set[str] = set()
    moves: list[dict[str, str]] = []
    grouped_paths: dict[str, list[str]] = {}
    skipped_paths: list[str] = []

    for source_rel_path in list(source_paths or []):
        clean_source = str(source_rel_path or "").strip()
        if not clean_source:
            continue
        source_abs = (notes_dir / clean_source).resolve()
        if not source_abs.exists() or not source_abs.is_file():
            skipped_paths.append(clean_source)
            continue

        category_label = str(path_category_map.get(clean_source, "") or "").strip()
        if not category_label:
            skipped_paths.append(clean_source)
            continue

        target_rel_path = _build_unique_target_rel_path(
            root_rel_path=session_root,
            category_label=category_label,
            source_rel_path=clean_source,
            planned_targets=planned_targets,
        )
        moves.append(
            {
                "source_rel_path": clean_source,
                "target_rel_path": target_rel_path,
                "category_label": category_label,
            }
        )
        grouped_paths.setdefault(category_label, []).append(clean_source)

    if not moves:
        return None, None

    if scope == "result_set":
        scope_text = "当前结果集"
    elif scope == "remaining":
        scope_text = "剩余未整理文件"
    else:
        scope_text = "整个知识库"
    preview_lines = [
        f"准备按当前分类整理{scope_text}中的 {len(moves)} 个文件：",
        f"- 目标根目录：{session_root}",
    ]
    for category_label, items in sorted(grouped_paths.items(), key=lambda item: (-len(item[1]), item[0])):
        preview_lines.append(f"- {category_label}：{len(items)} 个文件")
    if skipped_paths:
        preview_lines.append(f"- 跳过未归类文件：{len(skipped_paths)} 个")
    preview_lines.append("回复“确认整理”执行，或回复“取消”。")

    payload = {
        "scope": scope,
        "root_rel_path": session_root,
        "moves": moves,
        "skipped_paths": skipped_paths,
    }
    return "\n".join(preview_lines), payload
