from __future__ import annotations

import re
from pathlib import Path
from typing import Union

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


_FILENAME_NOISE_TOKENS = {
    "文件", "文档", "资料", "内容", "正文", "标题", "名称",
    "笔记", "记录", "整理", "总结", "备忘", "草稿", "副本", "版本", "最终", "临时", "测试",
    "新建", "文本", "文本文档",
    "the", "a", "an",
    "note", "notes", "doc", "docs", "document",
    "copy", "draft", "final", "new", "temp", "tmp", "test",
}

_FILENAME_TERM_ALIASES = {
    "工作经历": ("工作", "经历", "实习", "项目", "简历", "任职", "resume", "experience"),
    "简历": ("工作", "经历", "实习", "项目", "技能", "resume", "cv"),
}

_RESUME_HEADER_SIGNALS = (
    "resume", "curriculum vitae", "cv",
    "个人简历", "personal resume",
)

_RESUME_PROFILE_SIGNALS = (
    "姓名", "民族", "住址", "出生", "身高",
    "求职意向", "联系方式",
    "毕业院校", "教育背景", "专业", "主修课程",
    "兴趣爱好", "比赛经历", "自我评价",
)


def _is_filename_noise_term(term: str) -> bool:
    t = (term or "").strip().lower()
    if len(t) < 2:
        return True
    if t in _FILENAME_NOISE_TOKENS:
        return True
    if re.fullmatch(r"\d{2,8}", t):
        return True
    if re.fullmatch(r"v?\d+(?:\.\d+)*", t):
        return True
    if re.fullmatch(r"\d{4}[-_/]?\d{1,2}(?:[-_/]?\d{1,2})?", t):
        return True
    return False


def _extract_filename_focus_terms(path: str) -> list[str]:
    stem = Path(path).stem.lower()
    stem = re.sub(r"[_\-]+", " ", stem)
    chunks = re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fa5]{2,}", stem)

    terms: list[str] = []
    for chunk in chunks:
        parts = [chunk]
        if re.fullmatch(r"[\u4e00-\u9fa5]{3,}", chunk):
            parts.extend([x for x in re.split(r"[和与及的之在]", chunk) if x])

        for part in parts:
            token = part.strip().lower()
            if _is_filename_noise_term(token):
                continue
            if token not in terms:
                terms.append(token)

    return terms[:6]


def _expand_filename_term_variants(term: str) -> list[str]:
    t = (term or "").strip().lower()
    if not t:
        return []

    variants: list[str] = [t]
    variants.extend(_FILENAME_TERM_ALIASES.get(t, ()))

    if re.fullmatch(r"[\u4e00-\u9fa5]{4,}", t):
        variants.append(t[:2])
        variants.append(t[-2:])
        for i in range(len(t) - 1):
            variants.append(t[i:i + 2])

    deduped: list[str] = []
    for item in variants:
        token = (item or "").strip().lower()
        if _is_filename_noise_term(token):
            continue
        if token not in deduped:
            deduped.append(token)
    return deduped


def _term_has_content_evidence(term: str, search_space: str) -> bool:
    return any(v in search_space for v in _expand_filename_term_variants(term))


def _collect_signal_hits(search_space: str, keywords: tuple[str, ...]) -> list[str]:
    hits: list[str] = []
    for kw in keywords:
        if kw in search_space and kw not in hits:
            hits.append(kw)
    return hits


def _count_work_detail_markers(search_space: str) -> tuple[int, int, int]:
    date_range_patterns = [
        r"(?:19|20)\d{2}[./-]\d{1,2}\s*[-~至到]+\s*(?:19|20)\d{2}[./-]\d{1,2}",
        r"(?:19|20)\d{2}年\d{1,2}月?\s*[-~至到]+\s*(?:19|20)\d{2}年\d{1,2}月?",
    ]
    date_range_count = sum(len(re.findall(p, search_space)) for p in date_range_patterns)

    company_count = len(re.findall(r"[\u4e00-\u9fa5A-Za-z0-9]{2,}(?:有限公司|公司)", search_space))
    project_block_count = len(re.findall(r"(实习经历|工作经历|项目经验|项目经历)", search_space))
    return date_range_count, company_count, project_block_count


def _is_work_detail_dominant(search_space: str) -> bool:
    date_ranges, companies, project_blocks = _count_work_detail_markers(search_space)
    if date_ranges >= 3:
        return True
    if date_ranges >= 2 and companies >= 1:
        return True
    if date_ranges >= 1 and project_blocks >= 2:
        return True
    return False


def _collect_resume_signals(search_space: str) -> list[str]:
    profile_hits = _collect_signal_hits(search_space, _RESUME_PROFILE_SIGNALS)
    header_hits = _collect_signal_hits(search_space, _RESUME_HEADER_SIGNALS)

    merged: list[str] = []
    for token in profile_hits + header_hits:
        if token not in merged:
            merged.append(token)
    return merged


def _should_suggest_resume_name(search_space: str) -> bool:
    profile_hits = _collect_signal_hits(search_space, _RESUME_PROFILE_SIGNALS)
    header_hits = _collect_signal_hits(search_space, _RESUME_HEADER_SIGNALS)

    if not header_hits and len(profile_hits) < 3:
        return False

    # 工作明细型文档（大量时间区间+公司/项目段）默认视为“工作经历”可接受
    if _is_work_detail_dominant(search_space) and len(profile_hits) < 4:
        return False

    return len(profile_hits) >= 3 or (bool(header_hits) and len(profile_hits) >= 2)


def _is_work_experience_like_filename(path: str) -> bool:
    stem = Path(path).stem.lower()
    return ("工作经历" in stem) or ("工作" in stem and "经历" in stem)


def _is_resume_like_filename(path: str) -> bool:
    stem = Path(path).stem.lower()
    return ("简历" in stem) or ("resume" in stem) or ("cv" in stem)


def _answer_name_content_mismatch(repo_state) -> tuple[str, str]:
    path_to_shadow = {}
    for record in getattr(repo_state, "doc_records", []) or []:
        if not isinstance(record, dict):
            continue
        path = record.get("path")
        if path:
            path_to_shadow[path] = (record.get("shadow_tags", "") or "").lower()

    candidates = []
    naming_suggestions = []
    for path, doc in zip(repo_state.paths, repo_state.docs):
        text = (doc or "").lower()
        if len(text.strip()) < 80:
            continue

        filename_terms = _extract_filename_focus_terms(path)
        if not filename_terms:
            continue

        search_space = f"{text}\n{path_to_shadow.get(path, '')}"
        matched_terms = [term for term in filename_terms if _term_has_content_evidence(term, search_space)]
        single_specific_term = len(filename_terms) == 1 and len(filename_terms[0]) >= 4
        strict_mismatch = not matched_terms and (len(filename_terms) >= 2 or (single_specific_term and len(text) >= 80))

        if strict_mismatch:
            candidates.append({
                "path": path,
                "filename_terms": filename_terms,
            })
            continue

        if _is_work_experience_like_filename(path) and not _is_resume_like_filename(path) and _should_suggest_resume_name(search_space):
            signals = _collect_resume_signals(search_space)
            naming_suggestions.append({
                "path": path,
                "suggested_name": "简历",
                "signals": signals[:4],
            })

    if not candidates and not naming_suggestions:
        return (
            "我按“文件名关键词是否出现在正文/影子标签”做了快速核查，暂未发现明显不符的文件。"
            "\n如果你希望更严格，我可以按你指定的关键词再做一轮定向核对。",
            "name_content_mismatch",
        )

    lines: list[str] = []
    if candidates:
        show_n = min(10, len(candidates))
        lines.append(f"发现 {len(candidates)} 个疑似“文件名与内容不符”的文件（规则核查，建议人工复核）：")
        for i, item in enumerate(candidates[:show_n], 1):
            terms = "、".join(item["filename_terms"][:4])
            lines.append(f"{i}. {item['path']}（文件名关键词未命中：{terms}）")

        if len(candidates) > show_n:
            lines.append(f"...其余 {len(candidates) - show_n} 个未展开")

    if naming_suggestions:
        if lines:
            lines.append("")
        show_n = min(10, len(naming_suggestions))
        lines.append(f"另外发现 {len(naming_suggestions)} 个“内容基本匹配但命名可优化”的文件：")
        for i, item in enumerate(naming_suggestions[:show_n], 1):
            signal_text = "、".join(item["signals"]) if item["signals"] else "简历结构特征"
            lines.append(
                f"{i}. {item['path']}（更贴切命名建议：{item['suggested_name']}；依据：{signal_text}）"
            )
        if len(naming_suggestions) > show_n:
            lines.append(f"...其余 {len(naming_suggestions) - show_n} 个未展开")

    return "\n".join(lines), "name_content_mismatch"


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
    if topic == "size_consistency":
        return _answer_size_consistency(question, repo_state, last_user_question=last_user_question)
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
    if topic == "name_content_mismatch":
        return _answer_name_content_mismatch(repo_state)
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
