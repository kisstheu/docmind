from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.chat_text.core import (
    build_clean_merged_query,
    extract_strong_terms_from_question,
    extract_timeline_evidence_from_chunks,
    is_abstract_query,
    is_related_record_listing_request,
    is_result_expansion_followup,
    needs_timeline_evidence,
    normalize_colloquial_question,
    normalize_question_for_retrieval,
    redact_sensitive_text,
    strip_structured_request_words,
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("讲了啥？", "讲了什么？"),
        ("都是讲了啥？", "都是讲了什么？"),
        ("这是啥文件？", "这是啥文件？"),
    ],
)
def test_colloquial_content_normalization_keeps_its_constrained_scope(
    question,
    expected,
):
    assert normalize_colloquial_question(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (" 请帮我 看下 合同。 ", "合同"),
        ("麻烦展开说说 采购记录？", "采购记录"),
        ("交付\t\n验收?", "交付 验收"),
    ],
)
def test_retrieval_normalization_keeps_filler_punctuation_and_space_behavior(
    question,
    expected,
):
    assert normalize_question_for_retrieval(question) == expected


def test_structured_request_and_merged_query_normalization_stays_stable():
    assert strip_structured_request_words("请你 按时间线整理一下 合同吧") == "按 合同"
    assert build_clean_merged_query("请你 合同", "梳理一下 风险吧") == "合同 风险"


def test_sensitive_redaction_keeps_pattern_order_and_replacements():
    identity = "".join(("000000", "20000101", "000X"))
    mobile = "".join(("199", "0000", "0000"))
    email = "".join(("name", "@", "example.com"))
    long_number = "9" * 16
    text = f"证件 {identity} 手机 {mobile} 邮箱 {email} 编号 {long_number}。"

    assert redact_sensitive_text(text) == (
        "证件 [身份证号已脱敏] 手机 [手机号已脱敏] "
        "邮箱 [邮箱已脱敏] 编号 [长数字已脱敏]。"
    )


def test_strong_term_extraction_keeps_date_and_focus_term_order():
    question = "请看下2026年9月1日之后 08:30 的处理和法律性质？"

    assert extract_strong_terms_from_question(question) == [
        "2026年9月1日之后",
        "9月1日之后",
        "1日之后",
        "08:30",
        "法律性质",
        "性质",
        "处理",
        "之后",
    ]
    assert is_abstract_query(question) is False
    assert is_abstract_query("请帮我看下") is True


def test_timeline_evidence_keeps_first_seen_order_and_path_scoped_dedupe():
    repo_state = SimpleNamespace(
        chunk_paths=["合成甲.md", "合成甲.md", "合成乙.md"],
        chunk_texts=[
            "标题\n2026年9月1日 记录甲\n08:30 记录乙\n无日期",
            "2026年9月1日 记录甲\n9月2日 记录丙",
            "1日 记录丁\n08:30 记录乙",
        ],
    )

    assert extract_timeline_evidence_from_chunks([0, 1, 2, 0], repo_state) == [
        ("合成甲.md", "2026年9月1日 记录甲"),
        ("合成甲.md", "08:30 记录乙"),
        ("合成甲.md", "9月2日 记录丙"),
        ("合成乙.md", "1日 记录丁"),
        ("合成乙.md", "08:30 记录乙"),
    ]


def test_static_marker_predicates_keep_existing_boundaries():
    assert needs_timeline_evidence("请按时间线梳理") is True
    assert is_result_expansion_followup("扩大范围猜猜吧") is True
    assert is_result_expansion_followup("分析一下") is False
    assert is_related_record_listing_request("最近有哪些相关文档") is True
    assert is_related_record_listing_request("最近有哪些文档") is False
