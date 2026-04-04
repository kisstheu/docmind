from __future__ import annotations

import datetime
import re
from pathlib import Path
from uuid import uuid4

from infra.file_change_store import collect_file_snapshot

SUPPORTED_EXT = {
    ".txt", ".md", ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".ppt", ".pptx",
    ".png", ".jpg", ".jpeg", ".bmp", ".webp",
}
DEFAULT_TRASH_DIR = ".docmind_trash"

KW_DELETE = (
    "\u5220\u9664",  # 删除
    "\u5220\u6389",  # 删掉
    "\u5220\u4e86",  # 删了
    "\u79fb\u9664",  # 移除
    "\u53bb\u6389",  # 去掉
    "\u6254\u56de\u6536\u7ad9",  # 扔回收站
    "\u653e\u56de\u6536\u7ad9",  # 放回收站
)
KW_DELETE_HELP = (
    "\u600e\u4e48\u5220",  # 怎么删
    "\u5982\u4f55\u5220",  # 如何删
    "\u600e\u4e48\u5220\u9664",  # 怎么删除
    "\u5982\u4f55\u5220\u9664",  # 如何删除
    "\u80fd\u5220\u5417",  # 能删吗
    "\u53ef\u4ee5\u5220\u5417",  # 可以删吗
)
KW_CONFIRM = (
    "\u786e\u8ba4",  # 确认
    "\u786e\u8ba4\u5220\u9664",  # 确认删除
    "\u6267\u884c",  # 执行
    "\u6267\u884c\u5220\u9664",  # 执行删除
    "\u786e\u5b9a",  # 确定
    "\u662f",  # 是
    "yes",
    "y",
    "ok",
)
KW_CANCEL = (
    "\u53d6\u6d88",  # 取消
    "\u4e0d\u5220\u4e86",  # 不删了
    "\u7b97\u4e86",  # 算了
    "\u4e0d\u7528\u4e86",  # 不用了
    "\u5426",  # 否
    "\u4e0d\u662f",  # 不是
    "no",
    "n",
)


def _normalize_intent_text(text: str) -> str:
    q = (text or "").strip().lower()
    if not q:
        return ""
    q = q.replace(" ", "")
    q = q.replace("`", "").replace('"', "").replace("'", "")
    return q


def extract_explicit_filename(text: str) -> str | None:
    q = (text or "").strip()
    if not q:
        return None

    m = re.search(
        r"([A-Za-z0-9_\-\u4e00-\u9fa5\s]+?\.(?:txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx|png|jpg|jpeg|bmp|webp))",
        q,
        flags=re.IGNORECASE,
    )
    if not m:
        return None

    raw = re.sub(r"\s+", "", m.group(1).strip())
    raw = raw.strip("，。！？；,.!?;:")
    return raw or None


def is_delete_request(question: str) -> bool:
    q = _normalize_intent_text(question)
    if not q:
        return False

    if any(x in q for x in KW_DELETE_HELP):
        return False

    return any(k in q for k in KW_DELETE)


def parse_delete_confirmation_decision(question: str) -> str | None:
    q = _normalize_intent_text(question)
    if not q:
        return None

    if q in KW_CONFIRM or any(t in q for t in ("\u786e\u8ba4\u5220\u9664", "\u6267\u884c\u5220\u9664")):
        return "confirm"
    if q in KW_CANCEL or "\u53d6\u6d88" in q:
        return "cancel"
    return None


def resolve_delete_source_file(
    *,
    question: str,
    current_focus_file: str | None,
    last_result_set_items: list[str] | None,
) -> str | None:
    explicit = extract_explicit_filename(question)
    if explicit:
        return explicit

    if current_focus_file:
        return current_focus_file

    if last_result_set_items and len(last_result_set_items) == 1:
        return last_result_set_items[0]

    return None


def _build_trash_target_rel_path(source_rel_path: str, trash_dir_name: str = DEFAULT_TRASH_DIR) -> str:
    source_name = Path(source_rel_path).name
    date_folder = datetime.datetime.now().strftime("%Y%m%d")
    stamp = datetime.datetime.now().strftime("%H%M%S")
    suffix = f"__{stamp}_{uuid4().hex[:8]}"
    stem = Path(source_name).stem
    ext = Path(source_name).suffix
    target_name = f"{stem}{suffix}{ext}"
    return (Path(trash_dir_name) / date_folder / target_name).as_posix()


def build_delete_preview(
    *,
    notes_dir: Path,
    source_rel_path: str,
    trash_dir_name: str = DEFAULT_TRASH_DIR,
) -> tuple[str, dict] | tuple[None, None]:
    source_abs = (notes_dir / source_rel_path).resolve()
    if not source_abs.exists() or not source_abs.is_file():
        return None, None

    if source_abs.suffix.lower() not in SUPPORTED_EXT:
        return None, None

    target_rel_path = _build_trash_target_rel_path(source_rel_path, trash_dir_name=trash_dir_name)
    target_abs = (notes_dir / target_rel_path).resolve()

    before = collect_file_snapshot(source_abs, notes_dir)
    before_time = datetime.datetime.fromtimestamp(before["mtime"]).strftime("%Y-%m-%d %H:%M:%S")

    preview = (
        "Ready to soft-delete this file to trash. Confirm:\n"
        f"- source: {source_rel_path}\n"
        f"- trash: {target_rel_path}\n"
        f"- size: {before['size']} bytes\n"
        f"- mtime: {before_time}\n"
        f"- sha256: {before['sha256'][:16]}...\n"
        "Reply with: 确认删除 / cancel"
    )

    payload = {
        "source_rel_path": source_rel_path,
        "target_rel_path": target_rel_path,
        "target_abs_path": target_abs,
        "before_snapshot": before,
    }
    return preview, payload
