from __future__ import annotations

import re
from dataclasses import dataclass


def extract_result_set_from_answer(answer: str, entity_type: str = "文件") -> tuple[list[str], str]:
    items = re.findall(r"\d+\.\s*([^（]+?)\s*（", answer)
    items = [item.strip() for item in items if item.strip()]
    return items, entity_type


RESULT_SET_FOLLOWUP_PATTERNS = [
    r"^哪些是",
    r"^哪个是",
    r"^是哪个文件",
    r"^是哪个文档",
    r"^是哪个记录",
    r"^是在哪个文件",
    r"^是在哪个文档",
    r"^是在哪个记录",
    r"^在哪个文件",
    r"^在哪个文档",
    r"^在哪个记录",
    r"^在哪份文件",
    r"^在哪份文档",
    r"^在哪份记录",
    r"^哪几个是",
    r"^哪些属于",
    r"^哪些不是",
    r"^哪个不是",
    r"^其中哪些",
    r"^这里面哪些",
    r"^其中哪个",
    r"^这里面哪个",
    r"^哪些提到",
    r"^哪些涉及",
    r"^哪些和.*有关",
    r"^哪些与.*有关",
    r"^哪个最早",
    r"^哪个最新",
    r"^哪个最晚",
    r"^都有哪些",
    r"^有哪些",
    r"^分别有哪些",
    r"^分别是哪些",
    r"^具体有哪些",
    r"^具体是哪些",
    r"^都是什么",
    r"^分别是什么",
    r"^分别是关于什么",
    r"^分别是关于什么的",
    r"^各自是关于什么",
    r"^各自讲了什么",
    r"^分别讲了什么",
    r"^分别在说什么",
    r"^列一下",
    r"^展开说",
    r"^还有别的",
    r"^还有哪些",
    r"^哪几个",
    r"^哪几家",
    r"^目前知道的是哪几个",
    r"^目前知道的是哪几家",
    r"^所以.*哪几个",
    r"^所以.*哪几家",
    r"^这两个",
    r"^这两个是",
    r"^这两个其实",
    r"^是不是说这两个",
    r"^这两个.*相同",
    r"^这两个.*一致",
    r"^它们是",
    r"^它们.*相同",
    r"^它们.*一致",
    r"^也就是说",
    r"^也就是说其实",
    r"^其实是",
    r"^其实.*一样",
    r"^其实.*相同",
    r"^其实.*一致",
    r"^也就是说.*相同",
    r"^也就是说.*一致",
    r"^也就是说.*一样",
]
RESULT_SET_CONTINUATION_PATTERNS = [
    r"^更多$",
    r"^继续$",
    r"^再来$",
    r"^补充$",
    r"^还有吗",
    r"^还有没有",
    r"^还有别的",
    r"^还有其他",
    r"^还有哪些",
    r"^还有别的吗",
    r"^还有其他的吗",
    r"^还包括",
    r"^除此之外",
    r"^另外还有",
]


RESULT_SET_COMPARISON_TERMS = [
    "不同", "区别", "差异", "异同",
    "相同", "一样", "一致",
    "对比", "比较", "相比", "相较",
]


RESULT_SET_GROUP_REF_TERMS = [
    "这几个", "这几份", "这两个", "这三", "这两",
    "这些", "它们", "上述", "前面", "上面", "其中",
]

_CN_ORDINAL_DIGITS = dict(zip("零一二两三四五六七八九", (0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
_ORDINAL_TOKEN = r"[0-9一二两三四五六七八九十]+"
_FILE_ITEM_TARGET = r"(?:(?:个|份|篇)?(?:文件|文档|资料|记录)|(?:项|条))"
_SINGLE_FILE_REFERENCE = rf"第(?P<index>{_ORDINAL_TOKEN}){_FILE_ITEM_TARGET}"
_BARE_SINGLE_RESULT_REFERENCE = (
    rf"^(?:(?:那|那么|就|那就))?第(?P<index>{_ORDINAL_TOKEN})"
    r"(?:个)?(?:呢|怎么样|如何)?$"
)
_SELECTION_HELP = (
    "当前只支持选择单个文件或整个文件集合。"
    "请改为“第 N 个文件再展开”或“这些文件分别讲了什么”。"
)
_UNMAPPED_ORDINAL_HELP = (
    "我无法可靠确定这个序号对应哪个文件或对象。"
    "请明确要查看的文件名或对象名称。"
)
_ALL_FILE_REFERENCE_PATTERNS = (
    r"(?:这些|上述|前面|上面)(?:文件|文档|资料|记录|材料)",
    r"^(?:(?:请|帮我|麻烦|给我))?(?:(?:比较一下|对比一下|比较|对比))?(?:这些|它们|上述)",
    r"^(?:分别|各自|都)(?:讲|说|写|是|有|总结|概括|分析)",
    r"^(?:是)?(?:(?:关于|讲|说|写))?什么(?:内容|主题)?(?:的)?$",
    r"^(?:再)?(?:总结|概括)(?:一下|下|下吧|一下吧)?$",
    r"^(?:(?:请|帮我|给我))?(?:推荐|选择|挑选)(?:一|1)(?:个|份|项)$",
)


@dataclass(frozen=True)
class FileResultSetSelection:
    paths: tuple[str, ...] = ()
    rejection: str | None = None


def _parse_result_set_ordinal(token: str) -> int | None:
    value = (token or "").strip()
    if value.isdigit():
        return int(value)
    if value in _CN_ORDINAL_DIGITS:
        return _CN_ORDINAL_DIGITS[value]
    if value.count("十") == 1:
        tens, _, ones = value.partition("十")
        if (not tens or tens in _CN_ORDINAL_DIGITS) and (
            not ones or ones in _CN_ORDINAL_DIGITS
        ):
            return _CN_ORDINAL_DIGITS.get(tens, 1) * 10 + _CN_ORDINAL_DIGITS.get(ones, 0)
    return None


def _compact_result_set_question(question: str) -> str:
    return re.sub(r"[，。！？、,.!?；;：:\s]+", "", (question or ""))


def _extract_file_result_set_index(question: str) -> int | None:
    compact = _compact_result_set_question(question)
    match = re.search(_SINGLE_FILE_REFERENCE, compact)
    if not match:
        match = re.fullmatch(_BARE_SINGLE_RESULT_REFERENCE, compact)
    if not match:
        return None
    ordinal = _parse_result_set_ordinal(match.group("index"))
    return ordinal if ordinal and ordinal > 0 else None


def has_explicit_single_file_result_reference(question: str) -> bool:
    """Return whether the text explicitly selects one item from a file result set."""
    return (
        _extract_file_result_set_index(question) is not None
        and not _has_explicit_non_file_ordinal_target(question)
    )


def reject_unmapped_file_result_set_ordinal(
    question: str,
) -> FileResultSetSelection | None:
    """Reject an ordinal when retained file context is not the visible enumeration."""
    if not has_explicit_single_file_result_reference(question):
        return None
    return FileResultSetSelection(rejection=_UNMAPPED_ORDINAL_HELP)


def file_result_set_display_name(path: str) -> str:
    """Return a prompt-safe file name without exposing parent directories."""
    normalized = str(path or "").strip().replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def materialize_single_file_result_set_question(
    question: str,
    selected_file_path: str,
) -> str:
    """Replace a resolved ordinal reference with the selected file name."""
    if not has_explicit_single_file_result_reference(question):
        return question
    display_name = file_result_set_display_name(selected_file_path)
    if not display_name:
        return question

    replacement = f"文件《{display_name}》"
    full_reference = (
        rf"(?:(?:那|那么)\s*)?第\s*{_ORDINAL_TOKEN}\s*"
        rf"(?:(?:个|份|篇)?\s*(?:文件|文档|资料|记录)|(?:项|条))"
    )
    materialized, count = re.subn(full_reference, replacement, question, count=1)
    if count:
        return materialized

    bare_reference = rf"(?:(?:那|那么)\s*)?第\s*{_ORDINAL_TOKEN}\s*(?:个)?"
    return re.sub(bare_reference, replacement, question, count=1)


def _looks_like_unsupported_file_selection(question: str) -> bool:
    compact = _compact_result_set_question(question)
    if not compact:
        return False
    if re.search(rf"(?:前|最后){_ORDINAL_TOKEN}(?:个|份|篇|项|条)(?:文件|文档|资料|记录)?", compact):
        return True
    if re.search(rf"除了第{_ORDINAL_TOKEN}", compact):
        return True
    return len(re.findall(rf"第{_ORDINAL_TOKEN}(?:个|份|篇|项|条)?", compact)) >= 2


def _has_explicit_non_file_ordinal_target(question: str) -> bool:
    compact = _compact_result_set_question(question)
    return bool(
        re.search(
            rf"(?:(?:第|前|最后){_ORDINAL_TOKEN}(?:个)?)"
            r"(?:问题|章节|章|变化|版本|步骤)",
            compact,
        )
    )


def _looks_like_result_set_ordinal_correction(question: str) -> bool:
    compact = _compact_result_set_question(question)
    if not compact:
        return False
    if not any(term in compact for term in ("改成", "改为", "换成", "换为", "切成")):
        return False
    return _extract_file_result_set_index(question) is not None


def build_corrected_result_set_request(
    correction_question: str,
    previous_question: str | None,
    previous_answer: str | None,
) -> str | None:
    """Rebuild a rejected result-set request with the corrected ordinal."""
    if not _looks_like_result_set_ordinal_correction(correction_question):
        return None
    if not previous_question or not has_explicit_single_file_result_reference(previous_question):
        return None
    if not re.search(
        r"当前结果集中只有\s*\d+\s*个文件，请选择第\s*1[～~-]\d+\s*个",
        previous_answer or "",
    ):
        return None

    correction_match = re.search(
        _SINGLE_FILE_REFERENCE,
        _compact_result_set_question(correction_question),
    )
    if not correction_match:
        return None
    replacement = f"第{correction_match.group('index')}"
    return re.sub(
        rf"第\s*{_ORDINAL_TOKEN}",
        replacement,
        previous_question,
        count=1,
    )


def _looks_like_all_file_result_set_reference(question: str) -> bool:
    compact = _compact_result_set_question(question)
    if not compact:
        return False
    if any(term in compact for term in ("格式", "类型", "大小", "时间", "日期")):
        return False
    return any(re.search(pattern, compact) for pattern in _ALL_FILE_REFERENCE_PATTERNS)


def _looks_like_focused_content_reference(question: str) -> bool:
    compact = _compact_result_set_question(question)
    return bool(
        re.match(
            r"(?:这些|上述|前面|上面)"
            r"(?!(?:文件|文档|资料|记录|材料))",
            compact,
        )
    )


def resolve_file_result_set_selection(
    question: str,
    visible_paths: list[str] | tuple[str, ...],
    *,
    focus_file: str | None = None,
) -> FileResultSetSelection | None:
    candidates = tuple(
        str(item or "").strip()
        for item in visible_paths
        if str(item or "").strip()
    )
    if not candidates:
        return None

    if _has_explicit_non_file_ordinal_target(question):
        return None

    if _looks_like_unsupported_file_selection(question):
        return FileResultSetSelection(rejection=_SELECTION_HELP)

    ordinal = (
        _extract_file_result_set_index(question)
        if (
            has_explicit_single_file_result_reference(question)
            or _looks_like_result_set_ordinal_correction(question)
            or (focus_file is not None and looks_like_result_set_comparison_followup(question))
        )
        else None
    )
    if ordinal is not None:
        if focus_file and looks_like_result_set_comparison_followup(question):
            normalized_focus = re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "", focus_file.lower())
            focus_candidate = None
            for item in candidates:
                normalized_item = re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "", item.lower())
                if normalized_item == normalized_focus or normalized_item.endswith(normalized_focus) or normalized_focus.endswith(normalized_item):
                    focus_candidate = item
                    break
            if focus_candidate is not None:
                if ordinal > len(candidates):
                    return FileResultSetSelection(
                        rejection=(
                            f"当前结果集中只有 {len(candidates)} 个文件，"
                            f"请选择第 1～{len(candidates)} 个。"
                        )
                    )
                target = candidates[ordinal - 1]
                if focus_candidate == target:
                    return FileResultSetSelection(paths=(target,))
                return FileResultSetSelection(paths=(focus_candidate, target))
        if ordinal > len(candidates):
            return FileResultSetSelection(
                rejection=(
                    f"当前结果集中只有 {len(candidates)} 个文件，"
                    f"请选择第 1～{len(candidates)} 个。"
                )
            )
        return FileResultSetSelection(
            paths=(candidates[ordinal - 1],),
        )

    compact = _compact_result_set_question(question)
    counted_group = re.search(
        rf"这({_ORDINAL_TOKEN})(?:个|份|篇|项|条)",
        compact,
    )
    if counted_group:
        count = _parse_result_set_ordinal(counted_group.group(1))
        if count == len(candidates):
            return FileResultSetSelection(paths=candidates)
        return FileResultSetSelection(rejection=_SELECTION_HELP)

    if (
        _looks_like_all_file_result_set_reference(question)
        and not (focus_file and _looks_like_focused_content_reference(question))
    ):
        return FileResultSetSelection(paths=candidates)

    return None


def looks_like_result_set_followup(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False

    return (
        any(re.search(p, q) for p in RESULT_SET_FOLLOWUP_PATTERNS)
        or any(re.search(p, q) for p in RESULT_SET_CONTINUATION_PATTERNS)
    )


def has_selectable_result_set(
    items: list[str] | None,
    entity_type: str | None,
    answer_text: str | None,
    selectable: bool | None = None,
) -> bool:
    """Return whether the user was actually shown a non-empty numbered set."""
    if not items or not (entity_type or "").strip():
        return False
    if selectable is not None:
        return selectable

    numbered_items = []
    for line in (answer_text or "").splitlines():
        match = re.match(r"^\s*\d+[.、。)]\s*(.+?)\s*$", line)
        if match:
            numbered_items.append(match.group(1).strip())
    return bool(numbered_items)


def looks_like_result_set_continuation_followup(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    return any(re.search(p, q) for p in RESULT_SET_CONTINUATION_PATTERNS)


def looks_like_result_set_comparison_followup(question: str) -> bool:
    q = re.sub(r"\s+", "", (question or "").lower())
    if not q:
        return False

    if not any(term in q for term in RESULT_SET_COMPARISON_TERMS):
        return False

    if any(term in q for term in RESULT_SET_GROUP_REF_TERMS):
        return True

    return bool(re.search(r"(?:[一二两三四五六七八九十\d]+)(?:个|份|项|条)", q))


def last_turn_looks_like_enumeration(last_answer: str | None) -> bool:
    if not last_answer:
        return False

    text = last_answer.strip()

    numbered_lines = re.findall(r"(?:^|\n)\s*(?:\d+[.、]|[-*•])\s*", text)
    if len(numbered_lines) >= 2:
        return True

    if re.search(r"明确提到了\d+[家个项份条]", text):
        return True

    if re.search(r"提到了以下\d*[家个项份条]?", text):
        return True

    if re.search(r"以下\d+[家个项份条]", text):
        return True

    if "以下公司" in text or "提到了以下公司" in text:
        return True

    if "如下：" in text or "如下:" in text or "分别如下" in text:
        return True

    if text.count("有限公司") >= 2:
        return True

    return False


def infer_result_set_anchor(
    last_user_question: str | None,
    last_answer_type: str | None,
) -> str | None:
    prev = (last_user_question or "").strip()

    if last_answer_type == "enumeration_company":
        return "文档里提到的公司"
    if last_answer_type == "enumeration_file":
        return "上一轮提到的文件"
    if last_answer_type == "enumeration_person":
        return "上一轮提到的人物"

    if not prev:
        return None

    if "公司" in prev:
        return "文档里提到的公司"
    if "文件" in prev or "文档" in prev:
        return "上一轮提到的文件"
    if "人" in prev or "人物" in prev:
        return "上一轮提到的人物"

    return None


FILE_FOLLOWUP_STOP_TERMS = {
    "什么", "为何", "为什么", "怎么", "如何", "是否", "是不是",
    "哪个", "哪些", "哪几个", "几个", "三", "三个", "两", "两个", "几",
    "有啥", "有什么", "内容", "文件", "文档", "资料",
    "不同", "区别", "差异", "异同", "相同", "一样", "一致", "比较", "对比",
}


def _normalize_for_name_match(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "", (text or "").lower())


def _extract_question_focus_terms_for_files(question: str) -> list[str]:
    q = (question or "").lower()
    q = re.sub(r"[^\w\u4e00-\u9fa5]+", " ", q)
    raw_terms = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", q)

    terms: list[str] = []
    for term in raw_terms:
        t = term.strip()
        if not t or t in FILE_FOLLOWUP_STOP_TERMS:
            continue
        if re.fullmatch(r"\d+", t):
            continue
        if t not in terms:
            terms.append(t)
    return terms


def _narrow_result_set_files_by_question(items: list[str], question: str) -> list[str]:
    if not items:
        return items

    question_norm = _normalize_for_name_match(question)
    focus_terms = _extract_question_focus_terms_for_files(question)
    matched: list[str] = []
    for item in items:
        normalized_name = _normalize_for_name_match(item)
        # 1) 问题词直接命中文件名
        if focus_terms and any(term in normalized_name for term in focus_terms):
            matched.append(item)
            continue

        # 2) 反向匹配：文件名里的有效片段是否出现在问题中
        stem = re.sub(r"\.(txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx|png|jpg|jpeg|bmp|webp)$", "", item, flags=re.I)
        raw_parts = re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", stem.lower())
        parts: list[str] = []
        for p in raw_parts:
            if p in FILE_FOLLOWUP_STOP_TERMS or re.fullmatch(r"\d+", p):
                continue
            if p not in parts:
                parts.append(p)

        matched_by_part = False
        for part in parts:
            if part in question_norm:
                matched_by_part = True
                break

            if re.fullmatch(r"[\u4e00-\u9fa5]{4,}", part):
                for width in (2, 3):
                    for i in range(0, len(part) - width + 1):
                        sub = part[i:i + width]
                        if sub in FILE_FOLLOWUP_STOP_TERMS:
                            continue
                        if sub in question_norm:
                            matched_by_part = True
                            break
                    if matched_by_part:
                        break
            if matched_by_part:
                break

        if matched_by_part:
            matched.append(item)

    # 至少保留 2 个候选，避免过度收窄
    return matched if len(matched) >= 2 else items


def build_result_set_followup_query(
    question: str,
    last_user_question: str | None,
    last_answer_type: str | None,
    last_result_set_items: list[str] | None = None,
    last_result_set_entity_type: str | None = None,
) -> str:
    q = (question or "").strip()

    if last_result_set_items:
        entity_type = (last_result_set_entity_type or "项").strip()
        candidate_items = list(last_result_set_items)
        if entity_type == "文件":
            from app.chat_text.file_lookup import looks_like_file_set_content_question

            if looks_like_file_set_content_question(q):
                candidate_items = list(last_result_set_items)
            else:
                candidate_items = _narrow_result_set_files_by_question(candidate_items, q)
        item_text = "；".join(candidate_items[:20])

        if looks_like_result_set_continuation_followup(q):
            return (
                f"已知{entity_type}集合如下：{item_text}。"
                f"请继续在知识库中检索，判断是否还有其他符合条件的{entity_type}，并避免重复已知项。"
                f"当前追问：{q}"
            )

        if entity_type == "文件":
            return (
                f"已知文件如下：{item_text}。"
                f"请基于这些文件的内容回答：{q}"
            )

        return (
            f"候选{entity_type}如下：{item_text}。"
            f"请只在这些候选项中回答：{q}"
        )

    anchor = infer_result_set_anchor(last_user_question, last_answer_type)

    if not anchor:
        return q

    if q.startswith(("都有哪些", "有哪些", "分别有哪些", "分别是哪些", "具体有哪些", "具体是哪些", "都是什么")):
        return f"{anchor}分别有哪些？请列出完整名单。"

    if q.startswith(("哪些", "哪个", "哪几个")):
        return f"{anchor}中，{q}"

    if q.startswith(("其中", "这里面")):
        return f"{anchor}里，{q}"

    return f"基于{anchor}，回答：{q}"
