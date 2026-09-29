from __future__ import annotations

import pytest

from ai.prompt_builder import build_final_prompt
from ai.required_fact_delivery import (
    build_required_fact_delivery_contract,
    needs_required_fact_delivery,
)


MULTI_PART_CASES = [
    "合同对验收、付款和续约分别提出了什么要求？只按该文件回答。",
    "课程规则对作业、考试和补修分别有哪些限制？仅根据当前资料回答。",
    "采购说明中包装、运输和签收各自需要满足什么条件？",
]


@pytest.mark.parametrize("question", MULTI_PART_CASES)
def test_multi_part_request_gets_shared_required_fact_contract(question):
    prompt = build_final_prompt([], None, "", "合成参考事实。", question)

    assert needs_required_fact_delivery(question)
    assert "【共同适用必要事实完整性交付】" in prompt
    assert "不能因为它不对应某一个所求项名称而省略" in prompt
    assert "共同条件、时序、例外、前置要求或后续动作" in prompt
    assert "拆分呈现不得只把共享事实保留在其中一项" in prompt
    assert "相邻操作细节" in prompt
    assert "不把“应”或“可”改写为“必须”" in prompt


@pytest.mark.parametrize(
    "question",
    [
        "合同的验收日期是什么？",
        "比较两个方案的履行周期。",
        "列出资料中提到的全部风险。",
    ],
)
def test_adjacent_requests_do_not_enable_shared_required_fact_contract(question):
    assert not needs_required_fact_delivery(question)
    assert build_required_fact_delivery_contract(question) == ""
    assert "【共同适用必要事实完整性交付】" not in build_final_prompt(
        [], None, "", "合成参考事实。", question,
    )


def test_required_fact_delivery_contract_can_be_rolled_back(monkeypatch):
    monkeypatch.setenv("DOCMIND_REQUIRED_FACT_DELIVERY", "0")
    question = MULTI_PART_CASES[0]

    assert not needs_required_fact_delivery(question)
    assert "【共同适用必要事实完整性交付】" not in build_final_prompt(
        [], None, "", "合成参考事实。", question,
    )
