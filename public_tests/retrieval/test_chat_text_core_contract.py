from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.chat_text.core import (
    build_clean_merged_query,
    extract_strong_terms_from_question,
    extract_timeline_evidence_from_chunks,
    is_answer_depth_followup,
    is_abstract_query,
    is_related_record_listing_request,
    is_result_expansion_followup,
    merge_rewritten_query_with_strong_terms,
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
        (" 请帮我 看下 资料甲。 ", "资料甲"),
        ("麻烦展开说说 记录甲？", "记录甲"),
        ("主题甲\t\n描述?", "主题甲 描述"),
    ],
)
def test_retrieval_normalization_keeps_filler_punctuation_and_space_behavior(
    question,
    expected,
):
    assert normalize_question_for_retrieval(question) == expected


def test_structured_request_and_merged_query_normalization_stays_stable():
    assert strip_structured_request_words("请你 按时间线整理一下 资料甲吧") == "按 资料甲"
    assert build_clean_merged_query("请你 主题甲", "梳理一下 描述吧") == "主题甲 描述"


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


def test_strong_term_extraction_keeps_date_and_structural_term_order():
    question = "请看下2026年9月1日之后 08:30 的结果和状态？"

    assert extract_strong_terms_from_question(question) == [
        "2026年9月1日之后",
        "9月1日之后",
        "1日之后",
        "08:30",
        "之后",
    ]
    assert is_abstract_query(question) is False
    assert is_abstract_query("请帮我看下") is True


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("梳理一下时间线", ["时间线"]),
        ("之后发生了什么", ["之后"]),
        ("后来呢", ["后来"]),
        ("全过程", ["过程"]),
    ],
)
def test_structural_query_terms_remain_core_anchors(question, expected):
    assert extract_strong_terms_from_question(question) == expected
    assert is_abstract_query(question) is False


def test_date_query_terms_remain_core_anchors():
    question = "8月20日之后发生了什么？"

    assert extract_strong_terms_from_question(question) == [
        "8月20日之后",
        "20日之后",
        "之后",
    ]
    assert is_abstract_query(question) is False


@pytest.mark.parametrize(
    ("question", "rewritten_query", "expected"),
    [
        ("对象甲怎么描述的？", "对象甲 描述", "对象甲 描述"),
        ("事项甲是什么颜色？", "事项甲 颜色", "事项甲 颜色"),
        ("条目甲叫什么名称？", "条目甲 名称", "条目甲 名称"),
    ],
)
def test_ordinary_content_terms_remain_rewrite_content(
    question,
    rewritten_query,
    expected,
):
    assert extract_strong_terms_from_question(question) == []
    assert merge_rewritten_query_with_strong_terms(question, rewritten_query) == expected
    assert is_abstract_query(question) is True


@pytest.mark.parametrize(
    ("question", "rewritten_query", "expected"),
    [
        ("对象甲怎么描述的？", "怎么", "怎么"),
        ("事项甲是什么颜色？", "是什么", "是什么"),
        ("条目甲叫什么名称？", "叫什么", "叫什么"),
    ],
)
def test_ordinary_content_terms_are_not_promoted_as_structural_terms(
    question,
    rewritten_query,
    expected,
):
    assert merge_rewritten_query_with_strong_terms(question, rewritten_query) == expected


def test_ordinary_content_terms_are_not_structural_anchors():
    question = "描述、原因、颜色和名称分别是什么？"

    assert extract_strong_terms_from_question(question) == []
    assert is_abstract_query(question) is True


@pytest.mark.parametrize("question", ["详细点", "更详细"])
def test_answer_depth_followups_do_not_depend_on_strong_terms(question):
    assert extract_strong_terms_from_question(question) == []
    assert is_result_expansion_followup(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "再具体些",
        "可以再具体些吗？",
        "说得再具体一点",
        "能否更详细一些？",
        "展开说说",
        "再详细说说。",
    ],
)
def test_answer_depth_followup_recognizes_subjectless_natural_variants(question):
    assert is_answer_depth_followup(question) is True
    assert is_result_expansion_followup(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "具体有哪些文件？",
        "请详细分析新的合同议题。",
        "第二份采购资料再具体说明一下。",
    ],
)
def test_answer_depth_followup_rejects_subjectful_adjacent_requests(question):
    assert is_answer_depth_followup(question) is False


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
