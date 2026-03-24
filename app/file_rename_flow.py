from __future__ import annotations

import datetime
import re
from pathlib import Path

from infra.file_change_store import collect_file_snapshot

SUPPORTED_EXT = {".txt", ".md", ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".ppt", ".pptx"}
INVALID_FILENAME_CHARS = r'<>:"/\\|?*'


def _normalize_intent_text(text: str) -> str:
    q = (text or "").strip().lower()
    if not q:
        return ""
    q = q.replace(" ", "")
    q = q.replace("`", "").replace("\"", "").replace("'", "")
    return q


def _is_rename_history_query(question: str) -> bool:
    q = _normalize_intent_text(question)
    if not q:
        return False

    patterns = (
        r"当前.*(重命名|改名).*(过|记录|历史|日志|列表)",
        r"(重命名|改名).*(记录|历史|日志|列表|情况)",
        r"(有哪些|还有哪些|哪些).*(重命名|改名).*(过|了)?",
        r"(重命名|改名).*(过了|过吗|过没有)",
    )
    return any(re.search(p, q) for p in patterns)


def _is_rename_name_suggestion_query(question: str) -> bool:
    q = _normalize_intent_text(question)
    if not q:
        return False

    patterns = (
        r"(改成|叫|命名为|起名|取名).*(什么|啥|哪个|哪种).*(好|合适|贴切)",
        r"(改成|叫|命名为|起名|取名).*(什么|啥|哪个|哪种)",
        r"(什么|啥).*(名字|文件名).*(好|合适|贴切)",
        r"(怎么|如何).*(命名|起名|取名)",
        r"(建议|推荐).*(名字|文件名|命名)",
    )
    return any(re.search(p, q) for p in patterns)


def is_rename_history_query(question: str) -> bool:
    return _is_rename_history_query(question)


def is_rename_request(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False

    if _is_rename_history_query(q):
        return False

    if _is_rename_name_suggestion_query(q):
        return False

    patterns = [
        "重命名", "改名", "改一下", "改下", "改成", "命名为",
        "帮我改", "帮我改下", "帮我改一下",
    ]
    if any(p in q for p in patterns):
        return True

    if re.search(r"(那就|就)?叫.{1,40}吧", q):
        return True
    return False


def parse_confirmation_decision(question: str) -> str | None:
    q = (question or "").strip().lower()
    if not q:
        return None

    confirm_terms = {"确认", "确认重命名", "执行", "执行重命名", "确定", "是", "yes", "y", "ok", "好的"}
    cancel_terms = {"取消", "不改了", "算了", "不用了", "否", "不是", "no", "n"}

    if q in confirm_terms or any(t in q for t in ("确认重命名", "执行重命名")):
        return "confirm"
    if q in cancel_terms or "取消" in q:
        return "cancel"
    return None


def extract_explicit_filename(text: str) -> str | None:
    q = (text or "").strip()
    if not q:
        return None

    m = re.search(
        r"([A-Za-z0-9_\-\u4e00-\u9fa5\s]+?\.(?:txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx))",
        q,
        flags=re.IGNORECASE,
    )
    if not m:
        return None

    raw = re.sub(r"\s+", "", m.group(1).strip())
    raw = raw.strip("，。！？；：,!?;:")
    return raw or None


def _strip_tail_noise(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    t = t.strip("“”\"'`")
    t = re.split(r"[，。！？?!,;；]", t)[0].strip()
    t = re.sub(r"(可以帮我改.*|帮我改.*|行吗|好吗|可以吗|可不可以|吧)$", "", t).strip()
    t = t.strip("“”\"'`")
    return t


def extract_new_name_candidate(question: str) -> str | None:
    q = (question or "").strip()
    if not q:
        return None

    patterns = [
        r"(?:重命名为|改名为|命名为|改成)\s*[“\"']?([^“”\"'，。！？?!]+)",
        r"(?:那就|就)?叫\s*[“\"']?([^“”\"'，。！？?!]+?)(?:吧|$)",
    ]
    for p in patterns:
        m = re.search(p, q)
        if not m:
            continue
        candidate = _strip_tail_noise(m.group(1))
        if _is_rename_name_suggestion_query(candidate):
            continue
        if re.search(r"(什么|啥|哪个|哪种|怎么|如何|吗|呢|好呢|合适)", candidate):
            continue
        if candidate:
            return candidate

    return None


def normalize_target_filename(candidate: str, source_rel_path: str) -> str | None:
    name = (candidate or "").strip()
    if not name:
        return None

    name = name.strip("“”\"'`")
    name = name.replace("/", "").replace("\\", "")
    if any(c in name for c in INVALID_FILENAME_CHARS):
        return None

    source = Path(source_rel_path)
    parent = source.parent
    source_ext = source.suffix.lower()

    raw_target = Path(name)
    target_stem = raw_target.stem.strip()
    target_ext = raw_target.suffix.lower().strip()

    if not target_stem:
        return None

    if target_ext:
        if not target_ext.startswith("."):
            target_ext = "." + target_ext
        if target_ext not in SUPPORTED_EXT:
            return None
    else:
        target_ext = source_ext

    target_filename = f"{target_stem}{target_ext}"
    if target_filename == source.name:
        return source.as_posix()
    return (parent / target_filename).as_posix()


def resolve_source_file(
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


def build_rename_preview(
    *,
    notes_dir: Path,
    source_rel_path: str,
    target_rel_path: str,
) -> tuple[str, dict] | tuple[None, None]:
    source_abs = (notes_dir / source_rel_path).resolve()
    target_abs = (notes_dir / target_rel_path).resolve()

    if not source_abs.exists() or not source_abs.is_file():
        return None, None

    if target_abs.exists() and source_abs != target_abs:
        return None, None

    before = collect_file_snapshot(source_abs, notes_dir)
    before_time = datetime.datetime.fromtimestamp(before["mtime"]).strftime("%Y-%m-%d %H:%M:%S")

    preview = (
        "准备执行文件重命名，请确认：\n"
        f"- 原文件: {source_rel_path}\n"
        f"- 新文件名: {Path(target_rel_path).name}\n"
        f"- 新路径: {target_rel_path}\n"
        f"- 当前大小: {before['size']} bytes\n"
        f"- 当前修改时间: {before_time}\n"
        f"- 当前SHA256: {before['sha256'][:16]}...\n"
        "请回复“确认重命名”执行，或回复“取消”。"
    )

    payload = {
        "source_rel_path": source_rel_path,
        "target_rel_path": target_rel_path,
        "before_snapshot": before,
    }
    return preview, payload
