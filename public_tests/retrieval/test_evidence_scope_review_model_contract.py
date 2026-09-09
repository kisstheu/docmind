"""Opt-in semantic contracts; ordinary pytest never contacts a model service.

Run with DOCMIND_RUN_MODEL_CONTRACTS=1 and the application's existing credentials.
The offline companion tests verify local enforcement independently of the model.
"""
from __future__ import annotations

import os

import pytest

from ai.evidence_scope_review import review_generated_evidence_scope
from public_tests.state.test_evidence_scope_binding_contract import _candidate


# name, exact source chunks, generated claim, should pass review
MODEL_CASES = [
    ("product_cooccurrence", ["当前商品甲：价格60元。\n用户评价：以前买过一盒12个。"],
     "商品甲12个售价60元，所以5元/个，应优先购买。", False),
    ("product_unit_missing", ["当前商品甲：页面显示60元。"],
     "商品甲60元/个，购买40个总价2400元。", False),
    ("contract_history", ["当前套餐甲：费用1200元。\n历史备注：旧方案服务期12个月。"],
     "当前套餐甲每月费用100元。", False),
    ("device_variant", ["型号A当前售价900元。\n型号B规格：内存32GB。"],
     "型号A内存32GB，因此满足至少32GB的要求，建议选择A。", False),
    ("reversed_fields", ["历史备注：旧方案服务期12个月。\n当前套餐甲：费用1200元。"],
     "当前套餐甲每月费用100元。", False),
    ("different_chunks", ["当前套餐甲：费用1200元。", "历史备注：旧方案服务期12个月。"],
     "当前套餐甲每月费用100元。", False),
    ("conflicting_scope", ["对象甲本期服务费1200元，服务期12个月。", "修订通知：对象甲本期服务费改为2400元，其余不变。"],
     "对象甲本期平均费用100元/月，应据此优先选择甲。", False),
    ("product_explicit", ["商品甲：12支装，售价60元。"],
     "商品甲每支5元，由同一商品的60元包装价除以12支得到。", True),
    ("contract_explicit", ["套餐甲：服务期12个月，总价1200元。"],
     "套餐甲平均费用100元/月，由总价1200元除以服务期12个月得到。", True),
    ("device_explicit", ["型号A：内存32GB，当前售价900元。"],
     "型号A内存32GB，当前售价900元。", True),
    ("explicit_cross_chunk", ["套餐甲本期总额1200元；服务期限见条款R。", "条款R：套餐甲本期服务期限12个月。"],
     "套餐甲本期平均费用100元/月，依据总额及条款R的12个月期限。", True),
    ("conditional", ["商品甲：页面显示60元，计量基础待确认。"],
     "如果确认60元为每件价格，则40件为2400元；此前提尚未确认，这不是已确认总价，不能据此排名。", True),
    ("unknown_is_valid", ["商品甲：当前价格60元。\n历史评价：买过一盒12个。"],
     "页面显示60元，历史评价提及一盒12个，但两者对应关系未确认，暂不计算每个价格或据此推荐。", True),
    ("ordinary_qa", ["设备甲支持离线运行。"], "设备甲支持离线运行。", True),
    ("historical_qa", ["历史备注：旧方案服务期12个月。"], "旧方案的服务期是12个月。", True),
]


@pytest.mark.skipif(os.getenv("DOCMIND_RUN_MODEL_CONTRACTS") != "1", reason="explicit model opt-in required")
@pytest.mark.parametrize("name,texts,draft,expected", MODEL_CASES, ids=[c[0] for c in MODEL_CASES])
def test_model_relation_review(name, texts, draft, expected):
    from google import genai

    candidates = [_candidate(text, "合成资料.md", i) for i, text in enumerate(texts)]
    with genai.Client(api_key=os.environ["OPENAI_API_KEY"], vertexai=False) as client:
        review = review_generated_evidence_scope(
            answer_text=draft, question="请核对资料并回答。", source_candidates=candidates,
            client=client, model_id="gemini-2.5-flash",
        )
    assert not review.error, (name, review.error)
    assert review.verified is expected, (name, review.status, review.reasons)
    assert review.source_files == ("合成资料.md",)
