from __future__ import annotations

import re
from pathlib import Path


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
