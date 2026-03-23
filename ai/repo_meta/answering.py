from __future__ import annotations

import re

from ai.capability_common import (
    CATEGORY_CONFIRM_KEYWORDS,
    CATEGORY_KEYWORDS,
    CATEGORY_SUMMARY_KEYWORDS,
    LIST_FILE_KEYWORDS,
    TOTAL_SIZE_KEYWORDS,
    clean_text,
    contains_any,
    normalize_meta_question,
    CATEGORY_COUNT_KEYWORDS,
)

LIST_BY_TOPIC_PATTERNS = (
    r"列一下(.+?)的文件",
    r"列出(.+?)的文件",
    r"把(.+?)相关文件列出来",
    r"把(.+?)的文件列出来",
    r"(.+?)有哪些文件",
    r"(.+?)相关文档",
)

COUNT_KEYWORDS = (
    "多少文件", "多少个文件", "文件数量",
    "多少文档", "多少个文档", "文档数量",
    "有多少文件", "有多少文档",
    "目前有多少文件", "目前有多少文档",
    "现在有多少文件", "现在有多少文档",
    "总共有多少文件", "总共有多少文档",
)

FORMAT_KEYWORDS = (
    "哪些格式", "文件格式", "文档格式", "支持格式",
    "doc", "docx", "pdf", "txt", "md",
    "xls", "xlsx", "csv",
    "ppt", "pptx",
)

TIME_KEYWORDS = (
    "最近更新",
    "最近修改",
    "修改时间",
    "创建时间",
    "最新文件", "最早文件", "最晚文件",
    "最新文档", "最早文档", "最晚文档",
    "最新的文件", "最早的文件", "最晚的文件",
    "最新的文档", "最早的文档", "最晚的文档",
    "文件最新", "文件最早", "文件最晚",
    "文档最新", "文档最早", "文档最晚",
    "最新的两份", "最新的几份", "最早的两份", "最早的几份",
    "两份", "几份", "前两份", "前几份",
)

from ai.capability_common import format_bytes
from ai.repo_meta.category import (
    answer_repo_content_category_confirm_question,
    answer_repo_content_category_question,
    answer_repo_content_category_summary_question,
)
from ai.repo_meta.classifier import classify_repo_meta_question, extract_topic_from_list_request
from ai.repo_meta.semantic import find_files_by_semantic_cluster



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



def _answer_format(all_files) -> tuple[str, str]:
    suffixes = sorted({file.suffix.lower() or "[无后缀]" for file in all_files})
    answer = "当前知识库中的文件格式有：\n" + "\n".join(f"- {suffix}" for suffix in suffixes)
    return answer, "format"


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
from typing import Union, Tuple

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


def _answer_time(question: str, paths, file_times) -> tuple[str, str]:
    pairs = list(zip(paths, file_times))

    suffix_filter, label = _extract_suffix_filter(question)

    if suffix_filter:
        if isinstance(suffix_filter, tuple):
            pairs = [(p, t) for p, t in pairs if p.lower().endswith(suffix_filter)]
        else:
            pairs = [(p, t) for p, t in pairs if p.lower().endswith(suffix_filter)]

    if not pairs:
        if label:
            return f"当前没有找到 {label} 文档。", "time"
        if suffix_filter:
            return "当前没有找到符合条件的文档。", "time"
        return "当前知识库里还没有可用文档。", "time"

    q = (question or "").strip()
    top_k = _extract_top_k(q, default=1)

    sorted_latest = sorted(pairs, key=lambda x: x[1], reverse=True)
    sorted_earliest = sorted(pairs, key=lambda x: x[1])

    ask_latest = any(x in q for x in ["最新", "最近更新", "最近修改", "最晚"])
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


def _answer_list_files_by_topic(question: str, repo_state, model_emb=None, topic_summarizer=None) -> tuple[str, str]:
    target_topic = extract_topic_from_list_request(question)
    matched, _matched_tags, best_cluster = find_files_by_semantic_cluster(
        repo_state,
        model_emb,
        target_topic,
        topic_summarizer=topic_summarizer,
        limit=50,
    )

    if not matched:
        return f"当前知识库里暂时没有明显命中\"{target_topic}\"的文件。", "list_files_by_topic"

    label = best_cluster["label"] if best_cluster else "相关内容"
    lines = [f"按语义上最接近的类别（{label}）来看，相关文件大约有 {len(matched)} 个，先列出这些："]
    lines.extend(f"- {path}" for path in matched)
    return "\n".join(lines), "list_files_by_topic"



def answer_repo_meta_question(
    question: str,
    repo_state,
    model_emb=None,
    last_user_question: str | None = None,
    last_local_topic: str | None = None,
    topic_summarizer=None,
):
    paths = list(repo_state.paths)
    all_files = list(repo_state.all_files)
    file_times = list(repo_state.file_times)

    if not paths:
        return "当前知识库里还没有可用文档。", "empty"

    topic = classify_repo_meta_question(
        question,
        last_user_question=last_user_question,
        last_local_topic=last_local_topic,
    )

    if topic == "count":
        return _answer_count(paths)
    if topic == "total_size":
        return _answer_total_size(repo_state)
    if topic == "format":
        return _answer_format(all_files)
    if topic == "list_files_by_topic":
        return _answer_list_files_by_topic(
            question,
            repo_state,
            model_emb=model_emb,
            topic_summarizer=topic_summarizer,
        )
    if topic == "time":
        return _answer_time(question, paths, file_times)
    if topic == "list_files":
        return _answer_list_files(paths)
    if topic == "list_files_with_time":
        return _answer_list_files_with_time(paths, file_times)
    if topic == "category":
        return answer_repo_content_category_question(repo_state), topic
    if topic == "category_summary":
        return answer_repo_content_category_summary_question(repo_state, topic_summarizer=topic_summarizer), topic
    if topic == "category_confirm":
        return answer_repo_content_category_confirm_question(question, repo_state), topic

    return "我识别到你在问知识库的文件信息，但暂时没分清是数量、格式、时间还是列表。你可以换个更直接的问法。", "unknown_repo_meta"