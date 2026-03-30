from __future__ import annotations

import datetime
import re
from typing import Callable, Union

from ai.capability_common import clean_text, normalize_meta_question


def _extract_top_k(question: str, default: int = 1, max_k: int = 10) -> int:
    q = question or ""

    cn_num_map = {
        "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }

    m = re.search(r"哪(\d+)个", q)
    if m:
        return min(int(m.group(1)), max_k)

    for cn, num in cn_num_map.items():
        if f"哪{cn}个" in q or f"前{cn}个" in q or f"{cn}份" in q or f"前{cn}份" in q:
            return min(num, max_k)

    m = re.search(r"(\d+)份", q)
    if m:
        return min(int(m.group(1)), max_k)

    if "几个" in q or "哪些" in q:
        return min(3, max_k)

    return default


def _extract_suffix_filter(question: str) -> tuple[Union[str, tuple[str, ...], None], str | None]:
    q = (question or "").lower()

    family_map = [
        # 🔥 顺序很重要：精确 > 模糊
        ("docx", (".doc", ".docx"), "Word"),
        ("doc", (".doc", ".docx"), "Word"),
        ("word", (".doc", ".docx"), "Word"),

        ("pptx", (".ppt", ".pptx"), "PowerPoint"),
        ("ppt", (".ppt", ".pptx"), "PowerPoint"),
        ("powerpoint", (".ppt", ".pptx"), "PowerPoint"),

        ("xlsx", (".xls", ".xlsx"), "Excel"),
        ("xls", (".xls", ".xlsx"), "Excel"),
        ("excel", (".xls", ".xlsx"), "Excel"),

        ("pdf", (".pdf",), "PDF"),
        ("txt", (".txt",), "TXT"),
        ("md", (".md",), "Markdown"),
        ("csv", (".csv",), "CSV"),
    ]

    for key, suffixes, label in family_map:
        if key in q:
            # 单个后缀就返回字符串，多的返回 tuple
            if len(suffixes) == 1:
                return suffixes[0], label
            return suffixes, label

    return None, None


def _extract_explicit_date_filter(question: str) -> tuple[str | None, Callable[[datetime.datetime], bool] | None]:
    q = normalize_meta_question(clean_text(question))
    if not q:
        return None, None

    # yyyy-mm-dd / yyyy年m月d日 / yyyy/m/d
    full_date = re.search(
        r"(?<!\d)(?P<year>(?:19|20)\d{2})[年./\-](?P<month>\d{1,2})[月./\-](?P<day>\d{1,2})(?:日|号)?",
        q,
    )
    if full_date:
        year = int(full_date.group("year"))
        month = int(full_date.group("month"))
        day = int(full_date.group("day"))
        try:
            datetime.date(year, month, day)
        except ValueError:
            pass
        else:
            label = f"{year:04d}-{month:02d}-{day:02d}"
            return label, lambda dt: dt.year == year and dt.month == month and dt.day == day

    # m-d / m月d日
    month_day = re.search(
        r"(?<!\d)(?P<month>\d{1,2})[月./\-](?P<day>\d{1,2})(?:日|号)?",
        q,
    )
    if month_day:
        month = int(month_day.group("month"))
        day = int(month_day.group("day"))
        if 1 <= month <= 12 and 1 <= day <= 31:
            label = f"{month}月{day}日"
            return label, lambda dt: dt.month == month and dt.day == day

    # d号 / d日
    day_only = re.search(r"(?<!\d)(?P<day>\d{1,2})(?:日|号)(?!\d)", q)
    if day_only:
        day = int(day_only.group("day"))
        if 1 <= day <= 31:
            label = f"每月{day}日"
            return label, lambda dt: dt.day == day

    return None, None


def _answer_time(question: str, paths, file_times) -> tuple[str, str]:
    pairs = list(zip(paths, file_times))

    suffix_filter, label = _extract_suffix_filter(question)

    if suffix_filter:
        if isinstance(suffix_filter, tuple):
            pairs = [(p, t) for p, t in pairs if p.lower().endswith(suffix_filter)]
        else:
            pairs = [(p, t) for p, t in pairs if p.lower().endswith(suffix_filter)]

    q = (question or "").strip()
    date_label, date_matcher = _extract_explicit_date_filter(q)
    if date_matcher:
        pairs = [(p, t) for p, t in pairs if date_matcher(t)]

    if not pairs:
        if date_label and label:
            return f"当前没有找到日期为 {date_label} 的 {label} 文档。", "time"
        if date_label:
            return f"当前没有找到日期为 {date_label} 的文件。", "time"
        if label:
            return f"当前没有找到 {label} 文档。", "time"
        if suffix_filter:
            return "当前没有找到符合条件的文档。", "time"
        return "当前知识库里还没有可用文档。", "time"

    if date_label:
        sorted_by_time = sorted(pairs, key=lambda x: x[1])
        show_n = min(50, len(sorted_by_time))
        lines = [f"日期为 {date_label} 的文件共有 {len(sorted_by_time)} 个（按时间顺序）："]
        for i, (path, dt) in enumerate(sorted_by_time[:show_n], 1):
            lines.append(f"{i}. {path}（{dt.strftime('%Y-%m-%d %H:%M:%S')}）")
        if len(sorted_by_time) > show_n:
            lines.append(f"...其余 {len(sorted_by_time) - show_n} 个未展开")
        return "\n".join(lines), "time"

    top_k = _extract_top_k(q, default=1)

    sorted_latest = sorted(pairs, key=lambda x: x[1], reverse=True)
    sorted_earliest = sorted(pairs, key=lambda x: x[1])

    ask_latest = any(x in q for x in ["最近", "最新", "最近更新", "最近修改", "最晚"])
    ask_earliest = any(x in q for x in ["最早", "最旧"])
    ask_both = not ask_latest and not ask_earliest

    lines = []

    def _build_title(prefix: str, actual_k: int) -> str:
        if label:
            if actual_k < top_k:
                return f"当前只找到 {actual_k} 份 {label} 文档："
            return f"{prefix}的 {actual_k} 份 {label} 文档是："

        if suffix_filter:
            if actual_k < top_k:
                return f"当前只找到 {actual_k} 份符合条件的文档："
            return f"{prefix}的 {actual_k} 份符合条件的文档是："

        if actual_k < top_k:
            return f"当前只找到 {actual_k} 个文件："
        return f"{prefix}的 {actual_k} 个文件是："

    if ask_latest or ask_both:
        latest_items = sorted_latest[:top_k]
        actual_k = len(latest_items)

        lines.append(_build_title("最新", actual_k))
        for i, (path, dt) in enumerate(latest_items, 1):
            lines.append(f"{i}. {path}（{dt.strftime('%Y-%m-%d %H:%M:%S')}）")

    if ask_earliest or ask_both:
        earliest_items = sorted_earliest[:top_k]
        actual_k = len(earliest_items)

        if lines:
            lines.append("")
        lines.append(_build_title("最早", actual_k))
        for i, (path, dt) in enumerate(earliest_items, 1):
            lines.append(f"{i}. {path}（{dt.strftime('%Y-%m-%d %H:%M:%S')}）")

    return "\n".join(lines), "time"


def _answer_list_files(paths: list[str]) -> tuple[str, str]:
    show_n = min(50, len(paths))
    preview = "\n".join(f"- {path}" for path in paths[:show_n])
    if len(paths) > show_n:
        answer = (
            f"当前知识库里共有 {len(paths)} 个文件，先列出前 {show_n} 个：\n"
            f"{preview}\n- ...(其余 {len(paths) - show_n} 个未展开)"
        )
    else:
        answer = f"当前知识库里的文件如下：\n{preview}"
    return answer, "list_files"


def _answer_list_files_with_time(paths: list[str], file_times) -> tuple[str, str]:
    """列出所有文件并附带时间"""
    pairs = list(zip(paths, file_times))
    # 按时间倒序排列
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)

    show_n = min(50, len(pairs_sorted))
    lines = []
    for path, dt in pairs_sorted[:show_n]:
        lines.append(f"- {path}（{dt.strftime('%Y-%m-%d %H:%M:%S')}）")

    preview = "\n".join(lines)
    if len(paths) > show_n:
        answer = (
            f"当前知识库里共有 {len(paths)} 个文件（按时间倒序），先列出前 {show_n} 个：\n"
            f"{preview}\n- ...(其余 {len(paths) - show_n} 个未展开)"
        )
    else:
        answer = f"当前知识库里的文件如下（按时间倒序）：\n{preview}"
    return answer, "list_files_with_time"
