from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np

from retrieval.query_utils import extract_query_terms
from retrieval.search_term_scoring import (
    is_result_set_boilerplate_term,
    is_sentence_like_term,
)


_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
_FOLLOWUP_PREFIX_RE = re.compile(
    r"^(?:这些|那些|上述|它们|其中|分别|各自|全部|都|每个|每项|每条)*"
    r"(?:有没有|有无|是否|是不是|有)?"
)
_FOLLOWUP_SUFFIX_RE = re.compile(r"(?:吗|呢|呀|啊|么)+$")
_COMPARISON_SUFFIX_RE = re.compile(
    r"(?:以上|以下|以内|以外|之前|之后|前|后|左右|上下|起|止)+$"
)
_QUANTIFIED_UNIT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*([\u4e00-\u9fa5A-Za-z%‰℃°]{1,6})"
)
_SOURCE_ATTRIBUTE_MARKERS = (
    "来源", "署名", "出品方", "发布方", "制定方", "编制方",
    "谁出的", "哪家机构", "哪个机构", "所属机构", "发布机构", "制定机构",
    "编制机构", "出品机构",
)
_SOURCE_ACTION_MARKERS = ("发布", "制定", "编制", "出品")
_SOURCE_ACTOR_MARKERS = (
    "谁", "哪家", "哪个", "机构", "单位", "部门", "组织", "主体",
)
_TEMPORAL_ATTRIBUTE_MARKERS = ("日期", "时间", "年份", "年月")
_DOCUMENT_PROPERTY_MARKERS = (
    "标准", "规范", "指南", "规程", "制度", "办法", "合同", "协议",
    "报告", "通知", "公告", "手册", "说明书",
)
_DOCUMENT_PROPERTY_RELATIONS = (
    "可以认为是", "可认定为", "可视为", "是否属于", "是不是", "算不算",
    "能否算作", "可否视为", "属于", "算是", "算作", "都是", "是否", "算", "是",
)
_NON_PROPERTY_RELATIONS = ("符合", "满足", "达到", "依据", "按照", "参照")


def extract_requested_attribute_terms(question: str) -> tuple[str, ...]:
    """Extract the current turn's attribute signal without inherited entities."""
    compact = re.sub(r"[，。！？、,.!?；;：:\s]+", "", (question or "").strip())
    compact = _FOLLOWUP_PREFIX_RE.sub("", compact)
    compact = _FOLLOWUP_SUFFIX_RE.sub("", compact)

    terms: list[str] = []

    def append(term: str) -> None:
        normalized = (term or "").strip().lower()
        if normalized and normalized not in terms:
            terms.append(normalized)

    for raw_term in extract_query_terms(compact, compact):
        term = (raw_term or "").strip()
        if not term or term.isdigit():
            continue
        if is_result_set_boilerplate_term(term):
            continue
        if is_sentence_like_term(term) and term != compact:
            continue
        append(term)

    for match in _QUANTIFIED_UNIT_RE.finditer(compact):
        unit = _FOLLOWUP_SUFFIX_RE.sub("", match.group(1))
        unit = _COMPARISON_SUFFIX_RE.sub("", unit)
        if unit:
            # The first symbol after a quantity is the stable unit anchor; the
            # trailing comparison wording need not occur verbatim in sources.
            append(unit[0])

    return tuple(terms)


def classify_requested_attribute_kind(question: str) -> str | None:
    """Classify an extracted attribute independently from result-set scope."""
    attribute_text = "".join(extract_requested_attribute_terms(question))
    if not attribute_text:
        return None
    if any(marker in attribute_text for marker in _TEMPORAL_ATTRIBUTE_MARKERS):
        return None
    if _looks_like_document_property_judgment(attribute_text):
        return "document_property"
    if any(marker in attribute_text for marker in _SOURCE_ATTRIBUTE_MARKERS):
        return "source"
    if (
        any(marker in attribute_text for marker in _SOURCE_ACTION_MARKERS)
        and any(marker in attribute_text for marker in _SOURCE_ACTOR_MARKERS)
    ):
        return "source"
    if _looks_like_official_source_judgment(attribute_text):
        return "source"
    return None


def _looks_like_document_property_judgment(attribute_text: str) -> bool:
    """Keep provenance modifiers separate from the document class being judged."""
    if any(
        marker in attribute_text
        for marker in ("文件性质", "文档性质", "材料性质", "文件类型", "文档类型", "材料类型")
    ):
        return True

    for property_marker in _DOCUMENT_PROPERTY_MARKERS:
        marker_position = attribute_text.rfind(property_marker)
        if marker_position < 0:
            continue
        prefix = attribute_text[max(0, marker_position - 16):marker_position]
        if any(relation in prefix for relation in _NON_PROPERTY_RELATIONS):
            continue
        if any(relation in prefix for relation in _DOCUMENT_PROPERTY_RELATIONS):
            return True
    return False


def _looks_like_official_source_judgment(attribute_text: str) -> bool:
    if "官方" not in attribute_text:
        return False
    if any(marker in attribute_text for marker in _SOURCE_ACTION_MARKERS):
        return True
    return bool(
        re.fullmatch(
            r"(?:是否|是不是|都是|是|属于|算是)?"
            r"官方(?:的|文件|文档|资料|材料)?",
            attribute_text,
        )
    )


def build_requested_attribute_query(question: str) -> str:
    """Build a query for the requested dimension without importing adjacent evidence."""
    if classify_requested_attribute_kind(question) == "document_property":
        compact = re.sub(r"[，。！？、,.!?；;：:\s]+", "", question or "")
        terms = [marker for marker in _DOCUMENT_PROPERTY_MARKERS if marker in compact]
        return " ".join(dict.fromkeys(terms))
    return " ".join(extract_requested_attribute_terms(question))


def _normalized_token(text: str) -> str:
    return "".join(re.findall(r"[\w\u4e00-\u9fa5]", (text or "").lower()))


def _entity_text_affinity(normalized_entity: str, normalized_text: str) -> float:
    if not normalized_entity or not normalized_text:
        return 0.0
    if normalized_entity in normalized_text:
        return 1.0
    for width in range(len(normalized_entity) - 1, 1, -1):
        for start in range(0, len(normalized_entity) - width + 1):
            if normalized_entity[start:start + width] in normalized_text:
                return width / len(normalized_entity)
    return 0.0


def _encode_queries(model_emb, queries: Sequence[str]) -> np.ndarray | None:
    if not queries:
        return None
    encoded = np.asarray(
        model_emb.encode([_QUERY_PREFIX + query for query in queries]),
        dtype=float,
    )
    if encoded.ndim == 1:
        encoded = encoded.reshape(1, -1)
    if encoded.ndim != 2 or encoded.shape[0] != len(queries):
        return None
    return encoded


def select_entity_attribute_evidence_indices(
    *,
    question: str,
    entity_items: Sequence[str] | None,
    candidate_indices: Iterable[int],
    chunk_texts: Sequence[str],
    chunk_embeddings: np.ndarray,
    model_emb,
    max_seeds: int = 10,
) -> list[int]:
    """Select bounded joint entity/attribute evidence, not a fixed quota per entity."""
    entities = list(
        dict.fromkeys(
            item.strip()
            for item in (entity_items or ())
            if isinstance(item, str) and item.strip()
        )
    )[:20]
    if not entities or not (question or "").strip() or max_seeds <= 0:
        return []

    candidates = [
        index
        for index in dict.fromkeys(candidate_indices)
        if 0 <= index < len(chunk_texts) and index < len(chunk_embeddings)
    ]
    if not candidates:
        return []

    query_vectors = _encode_queries(
        model_emb,
        [question, *(f"{entity} {question}" for entity in entities)],
    )
    if query_vectors is None or query_vectors.shape[1] != chunk_embeddings.shape[1]:
        return []

    candidate_embeddings = chunk_embeddings[candidates]
    normalized_candidate_texts = [
        _normalized_token(chunk_texts[index] or "") for index in candidates
    ]
    lowercase_candidate_texts = [
        (chunk_texts[index] or "").lower() for index in candidates
    ]
    attribute_scores = candidate_embeddings @ query_vectors[0]
    joint_scores = candidate_embeddings @ query_vectors[1:].T
    attribute_terms = extract_requested_attribute_terms(question)

    selected: list[int] = []
    selected_set: set[int] = set()
    for entity_position, entity in enumerate(entities):
        normalized_entity = _normalized_token(entity)
        ranked: list[tuple[float, int, bool, bool]] = []
        for candidate_position, index in enumerate(candidates):
            exact_entity_hit = bool(
                normalized_entity
                and normalized_entity in normalized_candidate_texts[candidate_position]
            )
            entity_affinity = _entity_text_affinity(
                normalized_entity,
                normalized_candidate_texts[candidate_position],
            )
            if entity_affinity < 0.5:
                continue
            attribute_hit = any(
                term in lowercase_candidate_texts[candidate_position]
                for term in attribute_terms
            )
            score = (
                0.70 * float(joint_scores[candidate_position, entity_position])
                + 0.30 * float(attribute_scores[candidate_position])
                + 0.10 * entity_affinity
                + (0.10 if attribute_hit else 0.0)
            )
            ranked.append((score, index, attribute_hit, exact_entity_hit))

        if not ranked:
            continue
        exact_lexical_candidates = [
            item for item in ranked if item[2] and item[3]
        ]
        lexical_candidates = [item for item in ranked if item[2]]
        exact_semantic_candidates = [
            item for item in ranked if item[3] and item[0] >= 0.38
        ]
        semantic_candidates = [item for item in ranked if item[0] >= 0.38]
        eligible = (
            exact_lexical_candidates
            or lexical_candidates
            or exact_semantic_candidates
            or semantic_candidates
        )
        if not eligible:
            continue
        _score, best_index, _attribute_hit, _exact_entity_hit = max(
            eligible,
            key=lambda item: (item[0], -item[1]),
        )
        if best_index not in selected_set:
            selected.append(best_index)
            selected_set.add(best_index)
        if len(selected) >= max_seeds:
            break

    return selected


def select_scoped_file_attribute_evidence_indices(
    *,
    question: str,
    allowed_paths: Sequence[str] | None,
    candidate_indices: Iterable[int],
    chunk_paths: Sequence[str],
    chunk_texts: Sequence[str],
    chunk_embeddings: np.ndarray,
    model_emb,
) -> list[int]:
    """Select one attribute-focused evidence seed for every scoped file."""
    ordered_paths = list(
        dict.fromkeys(
            str(path or "").strip()
            for path in (allowed_paths or ())
            if str(path or "").strip()
        )
    )[:20]
    if not ordered_paths or not (question or "").strip():
        return []

    candidates = [
        index
        for index in dict.fromkeys(candidate_indices)
        if (
            0 <= index < len(chunk_paths)
            and index < len(chunk_texts)
            and index < len(chunk_embeddings)
        )
    ]
    query_vectors = _encode_queries(model_emb, [question])
    if (
        not candidates
        or query_vectors is None
        or query_vectors.shape[1] != chunk_embeddings.shape[1]
    ):
        return []

    attribute_scores = chunk_embeddings[candidates] @ query_vectors[0]
    attribute_terms = extract_requested_attribute_terms(question)
    selected: list[int] = []
    for path in ordered_paths:
        ranked: list[tuple[float, int]] = []
        for candidate_position, index in enumerate(candidates):
            if str(chunk_paths[index] or "").strip() != path:
                continue
            text = str(chunk_texts[index] or "").lower()
            lexical_bonus = 0.15 if any(term in text for term in attribute_terms) else 0.0
            ranked.append(
                (float(attribute_scores[candidate_position]) + lexical_bonus, index)
            )
        if ranked:
            selected.append(max(ranked, key=lambda item: (item[0], -item[1]))[1])
    return selected


__all__ = [
    "build_requested_attribute_query",
    "extract_requested_attribute_terms",
    "classify_requested_attribute_kind",
    "select_entity_attribute_evidence_indices",
    "select_scoped_file_attribute_evidence_indices",
]
