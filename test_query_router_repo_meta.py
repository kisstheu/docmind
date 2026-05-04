from __future__ import annotations

from ai.query_router import route_question


class _LoggerStub:
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
