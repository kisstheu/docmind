from __future__ import annotations

from app.chat_text.core import (
    build_clean_merged_query,
    build_timeline_evidence_text,
    extract_strong_terms_from_question,
    extract_timeline_evidence_from_chunks,
    is_abstract_query,
    is_related_record_listing_request,
    is_result_expansion_followup,
    keep_only_allowed_terms,
    merge_rewritten_query_with_strong_terms,
    needs_timeline_evidence,
    normalize_colloquial_question,
    normalize_question_for_retrieval,
    redact_sensitive_text,
    strip_structured_request_words,
)
from app.chat_text.file_lookup import maybe_build_file_location_answer
from app.chat_text.lookup_answer_main import maybe_build_direct_lookup_answer
from app.chat_text.related_records import (
    extract_related_topic,
    maybe_build_related_records_answer,
)

__all__ = [
    "build_clean_merged_query",
    "build_timeline_evidence_text",
    "extract_related_topic",
    "extract_strong_terms_from_question",
    "extract_timeline_evidence_from_chunks",
    "is_abstract_query",
    "is_related_record_listing_request",
    "is_result_expansion_followup",
    "keep_only_allowed_terms",
    "maybe_build_direct_lookup_answer",
    "maybe_build_file_location_answer",
    "maybe_build_related_records_answer",
    "merge_rewritten_query_with_strong_terms",
    "needs_timeline_evidence",
    "normalize_colloquial_question",
    "normalize_question_for_retrieval",
    "redact_sensitive_text",
    "strip_structured_request_words",
]
