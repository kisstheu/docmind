from __future__ import annotations

import datetime
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _normalize_intent_text(text: str) -> str:
    q = (text or "").strip().lower()
    if not q:
        return ""
    q = q.replace(" ", "")
    q = q.replace("`", "").replace("\"", "").replace("'", "")
    return q


def _is_image_path(path: str) -> bool:
    return Path((path or "").strip()).suffix.lower() in _IMAGE_EXTENSIONS


def _normalize_path_key(path: str) -> str:
    return (path or "").strip().replace("\\", "/").replace(" ", "").lower()


def _match_repo_path(ref: str, repo_paths: list[str]) -> str | None:
    key = _normalize_path_key(ref)
    if not key:
        return None

    for path in repo_paths:
        if _normalize_path_key(path) == key:
            return path

    for path in repo_paths:
        norm = _normalize_path_key(path)
        if norm.endswith(key) or key.endswith(norm):
            return path

    ref_name = Path((ref or "").strip().replace("\\", "/")).name
    ref_name_key = _normalize_path_key(ref_name)
    if not ref_name_key:
        return None
    for path in repo_paths:
        if _normalize_path_key(Path(path).name) == ref_name_key:
            return path
    return None


def _extract_explicit_image_filename(question: str) -> str | None:
    q = (question or "").strip()
    if not q:
        return None

    m = re.search(
        r"([A-Za-z0-9_\-\u4e00-\u9fa5\\/:.\s]+?\.(?:png|jpg|jpeg|bmp|webp))\b",
        q,
        flags=re.IGNORECASE,
    )
    if not m:
        return None

    raw = re.sub(r"\s+", " ", m.group(1).strip())
    raw = raw.strip("，。！？；:!?;[]【】()（）")
    return raw or None


_CN_NUM_MAP = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _parse_cn_number(token: str) -> int | None:
    t = (token or "").strip()
    if not t:
        return None
    if t.isdigit():
        return int(t)
    if t in _CN_NUM_MAP:
        return _CN_NUM_MAP[t]

    if len(t) == 2 and t[0] == "十" and t[1] in _CN_NUM_MAP:
        return 10 + _CN_NUM_MAP[t[1]]
    if len(t) == 2 and t[1] == "十" and t[0] in _CN_NUM_MAP:
        return _CN_NUM_MAP[t[0]] * 10
    if len(t) == 3 and t[1] == "十" and t[0] in _CN_NUM_MAP and t[2] in _CN_NUM_MAP:
        return _CN_NUM_MAP[t[0]] * 10 + _CN_NUM_MAP[t[2]]
    return None


def _extract_image_index(question: str) -> int | None:
    q = (question or "").strip()
    if not q:
        return None

    m = re.search(r"第\s*([0-9一二两三四五六七八九十]+)\s*(?:张|个|幅)?", q)
    if not m:
        return None
    return _parse_cn_number(m.group(1))


def is_image_view_request(question: str) -> bool:
    q = _normalize_intent_text(question)
    if not q:
        return False

    strong_terms = (
        "打开这张图",
        "打开这张图片",
        "打开图片",
        "打开图",
        "查看图片",
        "看图片",
        "看图",
        "预览图片",
        "预览图",
        "展示图片",
        "显示图片",
    )
    if any(term in q for term in strong_terms):
        return True

    if _extract_explicit_image_filename(question):
        return True

    if "打开第" in q and any(x in q for x in ("张", "个", "幅")):
        return True

    deictic = ("这张图", "这张图片", "这幅图", "该图", "这图")
    action = ("打开", "查看", "预览", "看")
    return any(d in q for d in deictic) and any(a in q for a in action)


def _build_image_list_tip(image_items: list[str], *, limit: int = 8) -> str:
    lines = ["当前结果集中有多张图片，请指定序号，例如“打开第1张图”："]
    for i, item in enumerate(image_items[:limit], start=1):
        lines.append(f"{i}. {item}")
    if len(image_items) > limit:
        lines.append(f"...（共 {len(image_items)} 张）")
    return "\n".join(lines)


def resolve_image_from_result_set(
    *,
    question: str,
    last_result_set_items: list[str] | None,
    current_focus_file: str | None,
    repo_paths: list[str],
) -> tuple[str | None, str | None]:
    if not last_result_set_items:
        return None, "当前还没有可用的文件结果集。请先让我定位到目标文件，再说“打开这张图”。"

    image_items: list[str] = []
    seen: set[str] = set()
    for item in last_result_set_items:
        if not _is_image_path(item):
            continue
        matched = _match_repo_path(item, repo_paths)
        if not matched:
            continue
        key = _normalize_path_key(matched)
        if key in seen:
            continue
        seen.add(key)
        image_items.append(matched)
    if not image_items:
        return None, "当前结果集中没有图片文件，暂时无法执行看图。"
    image_keys = {_normalize_path_key(x) for x in image_items}

    explicit = _extract_explicit_image_filename(question)
    if explicit:
        explicit_path = _match_repo_path(explicit, repo_paths)
        if explicit_path and _normalize_path_key(explicit_path) in image_keys:
            return explicit_path, None
        return None, "你指定的图片不在当前结果集中。请先把它查出来，或改用“打开第N张图”。"

    idx = _extract_image_index(question)
    if idx is not None:
        if idx < 1 or idx > len(image_items):
            return None, f"序号超出范围。当前可选 1 到 {len(image_items)}。"
        return image_items[idx - 1], None

    if current_focus_file:
        focused = _match_repo_path(current_focus_file, repo_paths)
        if focused and _normalize_path_key(focused) in image_keys:
            return focused, None

    if len(image_items) == 1:
        return image_items[0], None

    q = _normalize_intent_text(question)
    if any(x in q for x in ("这张图", "这张图片", "这幅图", "该图", "这图")):
        return image_items[0], None

    return None, _build_image_list_tip(image_items)


def create_shadow_image_copy(
    *,
    notes_dir: Path,
    source_rel_path: str,
    shadow_root: Path | None = None,
) -> tuple[Path | None, str | None]:
    source_abs = (notes_dir / source_rel_path).resolve()
    if not source_abs.exists() or not source_abs.is_file():
        return None, f"图片不存在或不可读：{source_rel_path}"

    root = shadow_root or (Path("logs") / "image_view_shadow")
    root.mkdir(parents=True, exist_ok=True)

    ext = source_abs.suffix.lower()
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source_abs.stem).strip("._") or "image"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    shadow_path = root / f"{stamp}_{safe_stem}{ext}"

    try:
        shutil.copy2(source_abs, shadow_path)
    except Exception as e:
        return None, f"创建影子副本失败：{e}"

    try:
        os.chmod(shadow_path, stat.S_IREAD)
    except Exception:
        # Viewer compatibility is higher priority; copy already isolates source.
        pass

    return shadow_path, None


def open_image_with_system_viewer(path: Path) -> tuple[bool, str | None]:
    try:
        if hasattr(os, "startfile"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True, None

        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.Popen([opener, str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, None
    except Exception as e:
        return False, str(e)
