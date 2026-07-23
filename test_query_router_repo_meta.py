from __future__ import annotations

import pytest

from ai.query_router import route_question
from ai.query_router import _get_smalltalk_rewrite_timeout_sec
from ai.repo_meta.classifier import classify_repo_meta_question
from ai.repo_meta.classifier_predicates import looks_like_time_request


class _LoggerStub:
    def debug(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


def test_inventory_listing_routes_to_repo_meta():
    result = route_question(
        "有哪些 文档？",
        "http://127.0.0.1:9/api/generate",
        "qwen2.5",
        _LoggerStub(),
    )
    assert result["route"] == "repo_meta"


def test_smalltalk_rewrite_timeout_should_default_to_four_seconds(monkeypatch):
    monkeypatch.delenv("DOCMIND_SMALLTALK_REWRITE_TIMEOUT", raising=False)
    assert _get_smalltalk_rewrite_timeout_sec() == 4.0


def test_smalltalk_rewrite_timeout_should_clamp_custom_value(monkeypatch):
    monkeypatch.setenv("DOCMIND_SMALLTALK_REWRITE_TIMEOUT", "30")
    assert _get_smalltalk_rewrite_timeout_sec() == 10.0


def test_file_locator_stays_normal_retrieval():
    result = route_question(
        "哪些文档里提到了某公司A？",
        "http://127.0.0.1:9/api/generate",
        "qwen2.5",
        _LoggerStub(),
    )
    assert result["route"] == "normal_retrieval"


def test_cross_domain_inventory_question_stays_out_of_scope():
    result = route_question(
        "有哪些餐厅？",
        "http://127.0.0.1:9/api/generate",
        "qwen2.5",
        _LoggerStub(),
    )
    assert result["route"] != "repo_meta"


def test_classifier_does_not_bypass_time_predicate():
    question = "最近合同到期吗？"

    predicate_result = looks_like_time_request(question)
    classifier_result = classify_repo_meta_question(question)

    assert predicate_result is False
    assert classifier_result != "time"


def test_bare_recent_followup_requires_repo_meta_context():
    assert classify_repo_meta_question("最近的呢？") != "time"


def test_explicit_file_creation_time_stays_time():
    assert classify_repo_meta_question("这份文档是什么时间创建的？") == "time"


@pytest.mark.parametrize(
    "question",
    [
        "最近的有哪些？",
        "最近时间有哪些？",
        "最近资料显示合同到期了吗？",
    ],
)
def test_classifier_requires_file_time_scope_or_context(question):
    assert classify_repo_meta_question(question) != "time"
