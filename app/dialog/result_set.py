from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path, PurePosixPath

from rich.console import Console
from rich.table import Table
from rich.text import Text


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
_BARE_SINGLE_RESULT_QUESTION_REFERENCE = (
    rf"^(?:(?:那|那么|就|那就))?第(?P<index>{_ORDINAL_TOKEN})(?:个)?"
    r"(?:的)?(?=.{1,24}$)(?=.{0,20}(?:什么|多少|几|怎么|如何|是否|哪|吗|呢)).+$"
)
_BARE_SINGLE_RESULT_DETAIL_REFERENCE = (
    rf"^(?:(?:那|那么|就|那就))?第(?P<index>{_ORDINAL_TOKEN})(?:个)?"
    r"(?:再)?(?:详细|具体|展开|深入)?(?:介绍|说明|讲讲|说说|分析)(?:一下|下)?$"
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
    display_item: str | None = None
    opaque_focus: str | None = None


@dataclass(frozen=True)
class GeneratedResultSetProvenance:
    display_items: tuple[str, ...] = ()
    source_candidates: tuple[str, ...] = ()
    source_hits: tuple[tuple[str, ...], ...] = ()
    opaque_focuses: tuple[str, ...] = ()
    entity_type: str | None = None
    enumeration_attempted: bool = False
    reliable: bool = False
    failure_reason: str | None = None


def structured_generated_enumeration_schema() -> dict[str, object]:
    """Return the model contract used for a Core-owned generated listing."""
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "display_name": {"type": "string"},
                        "source_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "evidence_text": {"type": "string"},
                    },
                    "required": [
                        "display_name",
                        "source_paths",
                        "evidence_text",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def build_structured_generated_enumeration_prompt(
    prompt: str,
    *,
    entity_type: str,
) -> str:
    target = str(entity_type or "").strip() or "对象"
    return (
        f"{prompt}\n\n"
        "【结构化枚举输出契约】\n"
        f"本轮要从参考片段列举“{target}”。只返回符合响应 Schema 的 JSON。\n"
        "items 的顺序就是最终展示顺序；不要输出解释性正文。\n"
        "每个 item 的 display_name 必须是材料中可逐字核对的稳定名称。\n"
        "source_paths 必须使用参考片段中 `文件【...】` 给出的完整路径，"
        "不得猜测、缩写或改写。\n"
        "evidence_text 必须逐字摘录能证明 display_name 与来源绑定的最小原文。\n"
        "无法可靠绑定的对象不要列入；没有可靠对象时返回空 items。"
    )


def _coerce_structured_generated_payload(payload) -> Mapping[str, object] | None:
    if isinstance(payload, Mapping):
        return payload
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, Mapping) else None
    if not isinstance(payload, str) or not payload.strip():
        return None
    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return decoded if isinstance(decoded, Mapping) else None


def _stable_generated_focus(
    entity_type: str,
    display_name: str,
    source_paths: tuple[str, ...],
) -> str:
    authority = json.dumps(
        {
            "entity": _normalize_generated_evidence(entity_type),
            "display": _normalize_generated_evidence(display_name),
            "sources": list(source_paths),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(authority.encode("utf-8")).hexdigest()[:24]
    return f"core-generated:v1:{digest}"


def materialize_structured_generated_result_set(
    payload,
    *,
    entity_type: str,
    candidate_paths: list[str] | tuple[str, ...] | None,
    repo_state,
) -> GeneratedResultSetProvenance:
    """Validate structured model output against bounded repository evidence."""
    target = str(entity_type or "").strip()
    decoded = _coerce_structured_generated_payload(payload)
    if not target or decoded is None:
        return GeneratedResultSetProvenance(
            entity_type=target or None,
            enumeration_attempted=True,
            failure_reason="invalid_payload",
        )

    raw_items = decoded.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        return GeneratedResultSetProvenance(
            entity_type=target,
            enumeration_attempted=True,
            failure_reason="empty_items",
        )

    repo_paths = list(getattr(repo_state, "paths", []) or [])
    repo_docs = list(getattr(repo_state, "docs", []) or [])
    if len(repo_paths) != len(repo_docs):
        return GeneratedResultSetProvenance(
            entity_type=target,
            enumeration_attempted=True,
            failure_reason="repository_alignment",
        )

    repo_documents: dict[str, str] = {}
    duplicate_repo_paths: set[str] = set()
    for raw_path, raw_document in zip(repo_paths, repo_docs):
        path = str(raw_path or "").strip()
        if not path or not isinstance(raw_document, str) or not raw_document.strip():
            continue
        if path in repo_documents:
            duplicate_repo_paths.add(path)
            continue
        repo_documents[path] = raw_document

    bounded_candidates = tuple(
        dict.fromkeys(
            str(path or "").strip()
            for path in candidate_paths or ()
            if (
                str(path or "").strip()
                and str(path or "").strip() in repo_documents
                and str(path or "").strip() not in duplicate_repo_paths
            )
        )
    )
    allowed_sources = set(bounded_candidates)
    if not allowed_sources:
        return GeneratedResultSetProvenance(
            entity_type=target,
            enumeration_attempted=True,
            failure_reason="missing_source_candidates",
        )

    display_items: list[str] = []
    source_hits: list[tuple[str, ...]] = []
    opaque_focuses: list[str] = []
    all_items_reliable = True
    seen_focuses: set[str] = set()

    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            all_items_reliable = False
            continue
        display_name = str(raw_item.get("display_name") or "").strip()
        evidence_text = str(raw_item.get("evidence_text") or "").strip()
        raw_sources = raw_item.get("source_paths")
        declared_sources = (
            tuple(
                dict.fromkeys(
                    str(path or "").strip()
                    for path in raw_sources
                    if str(path or "").strip()
                )
            )
            if isinstance(raw_sources, list)
            else ()
        )
        valid_sources = tuple(
            path for path in declared_sources if path in allowed_sources
        )

        item_reliable = bool(
            display_name
            and evidence_text
            and declared_sources
            and valid_sources == declared_sources
        )
        normalized_display = _normalize_generated_evidence(display_name)
        normalized_evidence = _normalize_generated_evidence(evidence_text)
        display_is_proven = False
        if item_reliable and normalized_display and normalized_evidence:
            for source_path in valid_sources:
                document = repo_documents[source_path]
                normalized_document = _normalize_generated_evidence(document)
                if normalized_evidence not in normalized_document:
                    item_reliable = False
                    break
                if normalized_display in normalized_evidence:
                    display_is_proven = True
        else:
            item_reliable = False
        item_reliable = item_reliable and display_is_proven

        display_items.append(display_name)
        source_hits.append(valid_sources if item_reliable else ())
        if item_reliable:
            focus = _stable_generated_focus(target, display_name, valid_sources)
            if focus in seen_focuses:
                item_reliable = False
                source_hits[-1] = ()
            else:
                seen_focuses.add(focus)
                opaque_focuses.append(focus)
        if not item_reliable:
            all_items_reliable = False

    reliable = bool(
        all_items_reliable
        and len(display_items) == len(raw_items)
        and len(source_hits) == len(display_items)
        and len(opaque_focuses) == len(display_items)
        and all(display_items)
    )
    return GeneratedResultSetProvenance(
        display_items=tuple(display_items),
        source_candidates=bounded_candidates,
        source_hits=tuple(source_hits),
        opaque_focuses=tuple(opaque_focuses) if reliable else (),
        entity_type=target,
        enumeration_attempted=True,
        reliable=reliable,
        failure_reason=None if reliable else "unreliable_item_binding",
    )


def render_structured_generated_result_set(
    provenance: GeneratedResultSetProvenance,
) -> str:
    if not provenance.display_items:
        return "没有识别出可可靠列举的对象。"
    lines: list[str] = []
    for index, display_name in enumerate(provenance.display_items, start=1):
        lines.append(f"{index}. {display_name}")
        sources = provenance.source_hits[index - 1] if index <= len(provenance.source_hits) else ()
        if sources:
            source_names = "、".join(file_result_set_display_name(path) for path in sources)
            lines.append(f"   来源文件：{source_names}")
    return "\n".join(lines)


def _extract_generated_display_items(answer_text: str) -> tuple[str, ...]:
    items: list[str] = []
    for raw_line in str(answer_text or "").splitlines():
        line = raw_line.strip()
        match = re.match(r"^(?:\d+[.、]|[-*•])\s*(.+?)\s*$", line)
        if match and match.group(1).strip():
            items.append(match.group(1).strip())
    return tuple(items)


def _normalize_generated_evidence(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).strip()
    normalized = re.sub(r"^\s*(?:#{1,6}|[-*•]|\d+[.、)])\s*", "", normalized)
    normalized = normalized.strip("`*_~'\"“”‘’《》【】[]()（） ")
    return re.sub(r"\s+", " ", normalized).casefold().strip()


def _document_exact_evidence_values(document_text: str) -> set[str]:
    values: set[str] = set()
    for raw_line in str(document_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        candidates = [line]
        if re.search(r"[:：]", line):
            _label, value = re.split(r"[:：]", line, maxsplit=1)
            if value.strip():
                candidates.append(value)
        if "|" in line:
            candidates.extend(cell for cell in line.split("|") if cell.strip())
        for candidate in candidates:
            normalized = _normalize_generated_evidence(candidate)
            if normalized:
                values.add(normalized)
    return values


def materialize_generated_result_set_provenance(
    answer_text: str,
    *,
    candidate_paths: list[str] | tuple[str, ...] | None,
    repo_state,
) -> GeneratedResultSetProvenance:
    """Prove display-item sources by exact evidence inside a bounded file set."""
    display_items = _extract_generated_display_items(answer_text)
    if not display_items:
        return GeneratedResultSetProvenance()

    repo_paths = list(getattr(repo_state, "paths", []) or [])
    repo_docs = list(getattr(repo_state, "docs", []) or [])
    if len(repo_paths) != len(repo_docs):
        return GeneratedResultSetProvenance(display_items=display_items)

    exact_repo_indices: dict[str, list[int]] = {}
    for index, raw_path in enumerate(repo_paths):
        path = str(raw_path or "").strip()
        if path:
            exact_repo_indices.setdefault(path, []).append(index)

    bounded_candidates: list[str] = []
    candidate_evidence: dict[str, set[str]] = {}
    for raw_candidate in candidate_paths or []:
        candidate = str(raw_candidate or "").strip()
        if not candidate or candidate in candidate_evidence:
            continue
        matching_indices = exact_repo_indices.get(candidate, [])
        if len(matching_indices) != 1:
            continue
        document = repo_docs[matching_indices[0]]
        if not isinstance(document, str) or not document.strip():
            continue
        bounded_candidates.append(candidate)
        candidate_evidence[candidate] = _document_exact_evidence_values(document)

    source_hits: list[tuple[str, ...]] = []
    for display_item in display_items:
        normalized_item = _normalize_generated_evidence(display_item)
        if not normalized_item:
            source_hits.append(())
            continue
        source_hits.append(
            tuple(
                path
                for path in bounded_candidates
                if normalized_item in candidate_evidence[path]
            )
        )

    return GeneratedResultSetProvenance(
        display_items=display_items,
        source_candidates=tuple(bounded_candidates),
        source_hits=tuple(source_hits),
    )


def resolve_generated_result_set_selection(
    question: str,
    display_items: list[str] | tuple[str, ...] | None,
    source_hits: list[list[str]] | tuple[tuple[str, ...], ...] | None,
    source_candidates: list[str] | tuple[str, ...] | None,
    opaque_focuses: list[str] | tuple[str, ...] | None = None,
) -> FileResultSetSelection | None:
    """Resolve an ordinal only when its visible item has one proven backing file."""
    if not display_items:
        return None
    if _has_explicit_non_file_ordinal_target(question):
        return None
    ordinal = _extract_file_result_set_index(question)
    if ordinal is None:
        return None

    if not source_hits or len(source_hits) != len(display_items):
        return FileResultSetSelection(rejection=_UNMAPPED_ORDINAL_HELP)
    if ordinal > len(display_items):
        return FileResultSetSelection(
            rejection=(
                f"当前生成结果中只有 {len(display_items)} 个条目，"
                f"请选择第 1～{len(display_items)} 个。"
            )
        )

    allowed_sources = {
        str(path or "").strip()
        for path in source_candidates or []
        if str(path or "").strip()
    }
    hits = tuple(
        dict.fromkeys(
            str(path or "").strip()
            for path in source_hits[ordinal - 1]
            if str(path or "").strip() in allowed_sources
        )
    )
    focuses = tuple(str(focus or "").strip() for focus in opaque_focuses or ())
    has_materialized_focuses = len(focuses) == len(display_items) and all(focuses)
    if not hits or (not has_materialized_focuses and len(hits) != 1):
        return FileResultSetSelection(rejection=_UNMAPPED_ORDINAL_HELP)
    return FileResultSetSelection(
        paths=hits,
        display_item=str(display_items[ordinal - 1] or "").strip() or None,
        opaque_focus=focuses[ordinal - 1] if has_materialized_focuses else None,
    )


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
        match = re.fullmatch(_BARE_SINGLE_RESULT_QUESTION_REFERENCE, compact)
    if not match:
        match = re.fullmatch(_BARE_SINGLE_RESULT_DETAIL_REFERENCE, compact)
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


def format_file_result_set_size(num_bytes: int) -> str:
    """Format a file size with deterministic binary units for result-set details."""
    size = max(0, int(num_bytes))
    if size < 1024:
        return f"{size} B"
    for unit, divisor in (
        ("KiB", 1024),
        ("MiB", 1024**2),
        ("GiB", 1024**3),
    ):
        if unit == "GiB" or size < divisor * 1024:
            return f"{size / divisor:.1f} {unit}"
    raise AssertionError("unreachable")


def _normalize_file_result_set_identity(path: object) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    normalized = re.sub(r"/+", "/", normalized)
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _safe_notes_relative_path(path: object, notes_dir: Path) -> str | None:
    normalized = _normalize_file_result_set_identity(path)
    if not normalized:
        return None

    root = Path(os.path.abspath(notes_dir))
    is_windows_absolute = bool(re.match(r"^[A-Za-z]:/", normalized))
    if is_windows_absolute and Path(normalized).anchor == "":
        return None

    if normalized.startswith("/") or is_windows_absolute:
        candidate = Path(normalized)
    else:
        relative = PurePosixPath(normalized)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        candidate = root.joinpath(*relative.parts)

    try:
        absolute_candidate = Path(os.path.abspath(candidate))
        return absolute_candidate.relative_to(root).as_posix()
    except (OSError, ValueError):
        return None


def _unique_identity_indices(paths: list[object], notes_dir: Path) -> dict[str, int]:
    indices: dict[str, int] = {}
    duplicates: set[str] = set()
    for index, path in enumerate(paths):
        identity = _safe_notes_relative_path(path, notes_dir)
        if not identity:
            continue
        if identity in indices:
            duplicates.add(identity)
        else:
            indices[identity] = index
    for identity in duplicates:
        indices.pop(identity, None)
    return indices


def _coerce_file_result_set_size(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        size = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return size if size >= 0 else None


def _coerce_file_result_set_time(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return datetime.fromtimestamp(value)
        except (OSError, OverflowError, ValueError):
            return None
    return None


def _result_set_metadata_for_item(
    item: str,
    *,
    repo_state,
    notes_dir: Path,
    path_indices: dict[str, int],
    basename_indices: dict[str, list[int]],
) -> tuple[int | None, int | None, datetime | None, str | None]:
    relative_path = _safe_notes_relative_path(item, notes_dir)
    repo_index = path_indices.get(relative_path or "")
    if repo_index is None:
        basename = file_result_set_display_name(item).casefold()
        matches = basename_indices.get(basename, [])
        if len(matches) == 1:
            repo_index = matches[0]

    if repo_index is None:
        return None, None, None, relative_path

    repo_paths = list(getattr(repo_state, "paths", []) or [])
    repo_identity = _safe_notes_relative_path(repo_paths[repo_index], notes_dir)
    if not repo_identity:
        return None, None, None, relative_path
    relative_path = repo_identity

    matching_records = []
    for record in getattr(repo_state, "doc_records", []) or []:
        if not isinstance(record, dict):
            continue
        if _safe_notes_relative_path(record.get("path"), notes_dir) == repo_identity:
            matching_records.append(record)
    record = matching_records[0] if len(matching_records) == 1 else None

    size = None
    modified_time = None
    if record is not None:
        for key in ("file_size", "size", "bytes"):
            size = _coerce_file_result_set_size(record.get(key))
            if size is not None:
                break
        modified_time = _coerce_file_result_set_time(record.get("file_time"))

    file_times = list(getattr(repo_state, "file_times", []) or [])
    if modified_time is None and repo_index < len(file_times):
        modified_time = _coerce_file_result_set_time(file_times[repo_index])

    stat_path = None
    all_files = list(getattr(repo_state, "all_files", []) or [])
    if repo_index < len(all_files):
        all_file_relative = _safe_notes_relative_path(all_files[repo_index], notes_dir)
        if all_file_relative == repo_identity:
            stat_path = Path(all_files[repo_index])
    if stat_path is None:
        stat_path = Path(notes_dir).joinpath(*PurePosixPath(repo_identity).parts)

    if size is None or modified_time is None:
        try:
            stat_result = stat_path.stat()
        except (OSError, ValueError):
            stat_result = None
        if stat_result is not None:
            if size is None:
                size = stat_result.st_size
            if modified_time is None:
                modified_time = datetime.fromtimestamp(stat_result.st_mtime)

    return repo_index, size, modified_time, relative_path


def build_file_result_set_metadata_detail(
    question: str,
    items: list[str] | tuple[str, ...] | None,
    *,
    entity_type: str | None,
    selectable: bool | None,
    repo_state,
    notes_dir: Path,
) -> str | None:
    """Render metadata for the active file result set without changing its order."""
    normalized_question = re.sub(
        r"[。！？!?]+$",
        "",
        str(question or "").strip(),
    ).strip()
    if (
        normalized_question != "显示详情"
        or not items
        or entity_type != "文件"
        or selectable is not True
    ):
        return None

    ordered_items = list(items)
    repo_paths = list(getattr(repo_state, "paths", []) or [])
    path_indices = _unique_identity_indices(repo_paths, Path(notes_dir))
    basename_indices: dict[str, list[int]] = {}
    for index, path in enumerate(repo_paths):
        basename = file_result_set_display_name(path).casefold()
        basename_indices.setdefault(basename, []).append(index)

    display_name_counts: dict[str, int] = {}
    for item in ordered_items:
        name = file_result_set_display_name(item).casefold()
        display_name_counts[name] = display_name_counts.get(name, 0) + 1

    table = Table(
        box=None,
        collapse_padding=True,
        padding=(0, 1),
        show_edge=False,
    )
    table.add_column("#", no_wrap=True)
    table.add_column("类型", no_wrap=True)
    table.add_column("大小", no_wrap=True)
    table.add_column("修改时间", no_wrap=True)
    table.add_column("文件")
    for index, item in enumerate(ordered_items, 1):
        display_name = file_result_set_display_name(item) or "未知"
        _repo_index, size, modified_time, relative_path = _result_set_metadata_for_item(
            item,
            repo_state=repo_state,
            notes_dir=Path(notes_dir),
            path_indices=path_indices,
            basename_indices=basename_indices,
        )
        extension = PurePosixPath(display_name).suffix.lower() or "未知"
        size_text = format_file_result_set_size(size) if size is not None else "未知"
        time_text = (
            modified_time.strftime("%Y-%m-%d %H:%M:%S")
            if modified_time is not None
            else "未知"
        )

        show_relative_path = bool(
            relative_path
            and (
                "/" in relative_path
                or display_name_counts.get(display_name.casefold(), 0) > 1
            )
        )
        file_text = relative_path if show_relative_path else display_name
        table.add_row(str(index), extension, size_text, time_text, Text(file_text))

    output = StringIO()
    Console(
        file=output,
        force_terminal=False,
        color_system=None,
        highlight=False,
        width=120,
    ).print(table)
    return f"当前文件结果集详情：\n\n{output.getvalue().rstrip()}"


def sort_file_result_set_by_filename(
    question: str,
    items: list[str] | tuple[str, ...] | None,
    *,
    entity_type: str | None,
    selectable: bool | None,
) -> list[str] | None:
    """Sort an active selectable file result set by its visible file name."""
    normalized_question = re.sub(
        r"[。！？!?]+$",
        "",
        str(question or "").strip(),
    ).strip()
    if (
        normalized_question != "按文件名排列"
        or not items
        or entity_type != "文件"
        or selectable is not True
    ):
        return None

    return sorted(
        list(items),
        key=lambda path: (file_result_set_display_name(path), str(path)),
    )


def render_file_result_set_filename_sort(items: list[str] | tuple[str, ...]) -> str:
    """Render every sorted item using its user-visible file name."""
    lines = ["已按文件名排列："]
    lines.extend(
        f"{index}. {file_result_set_display_name(path)}"
        for index, path in enumerate(items, 1)
    )
    return "\n".join(lines)


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


def materialize_generated_result_set_item_question(
    question: str,
    display_item: str,
) -> str:
    """Replace an ordinal with a validated opaque collection item's label."""
    if not has_explicit_single_file_result_reference(question):
        return question
    label = str(display_item or "").strip()
    if not label:
        return question
    replacement = f"对象《{label}》"
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
            from app.chat_text.file_lookup import (
                looks_like_all_items_file_set_content_question,
            )

            if looks_like_all_items_file_set_content_question(q):
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
