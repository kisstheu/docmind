from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ai.evidence_scope import build_evidence_scope_contract
from ai.evidence_scope_review import (
    parse_evidence_scope_review, render_unverified_evidence, review_generated_evidence_scope,
)
from app.dialog.state_machine import ConversationState
from public_tests.state.test_repo_meta_generic_file_list_scope_contract import _run_turns
from retrieval.search_context import CanonicalSourceCandidate, canonical_source_candidate_id


def _candidate(text, path="材料甲.md", chunk=0):
    return CanonicalSourceCandidate(
        canonical_source_candidate_id(path=path, chunk_id=chunk, start=0, end=len(text), text=text),
        chunk, path, chunk, 0, len(text), text,
    )


def _ref(candidate):
    return {"source_id": candidate.source_id, "line_start": 1, "line_end": len(candidate.text.splitlines())}


def _review_payload(candidates, reason=None):
    return json.dumps({
        "status": "UNVERIFIED" if reason else "VERIFIED",
        "bindings": [] if reason else [{
            "claim": candidates[0].text, "assertion": "CONFIRMED", "basis": "EXPLICIT",
            "evidence": [_ref(candidates[0])],
        }],
        "issues": [{"reason": reason, "evidence": [_ref(c) for c in candidates]}] if reason else [],
        "support": [_ref(c) for c in candidates],
    }, ensure_ascii=False)


CASES = [
    ("商品", "价格60元。\n用户评价：以前买过一盒12个。", "5元/个", "relation"),
    ("合同", "当前套餐费用1200元。\n历史备注：旧方案服务期12个月。", "100元/月", "qualifier"),
    ("设备", "型号A当前售价900元。\n型号B规格：内存32GB。", "型号A内存32GB", "subject"),
]


@pytest.mark.parametrize("domain,text,bad_fact,reason", CASES)
@pytest.mark.parametrize("reverse", [False, True])
def test_rejection_delivers_only_exact_scoped_evidence(domain, text, bad_fact, reason, reverse):
    parts = text.splitlines()
    candidates = [_candidate(line, f"合成{domain}.md", i) for i, line in enumerate(parts)]
    if reverse:
        candidates.reverse()
    review = parse_evidence_scope_review(_review_payload(candidates, reason), candidates)
    assert not review.verified and not review.error
    answer = render_unverified_evidence(review)
    assert bad_fact not in answer
    assert "暂缓据此选择" in answer
    assert all(c.text in answer and c.path in answer for c in candidates)
    assert review.source_files == (f"合成{domain}.md",)


@pytest.mark.parametrize("text", [
    "合成商品甲：12支装，售价60元。",
    "合成合同甲：服务期12个月，总价1200元。",
    "合成设备型号A：内存32GB，售价900元。",
])
def test_explicit_local_scope_can_be_verified(text):
    candidate = _candidate(text)
    review = parse_evidence_scope_review(_review_payload([candidate]), [candidate])
    assert review.verified and review.evidence[0].quote == text


@pytest.mark.parametrize("other_path", ["材料甲.md", "材料乙.md"])
def test_explicit_cross_chunk_or_source_relation_is_not_rejected(other_path):
    candidates = [
        _candidate("对象甲本期总额1200元，适用期限见条款R。"),
        _candidate("条款R：对象甲本期期限为12个月。", other_path, 1),
    ]
    assert parse_evidence_scope_review(_review_payload(candidates), candidates).verified


@pytest.mark.parametrize("reason", ["unit", "conflict", "recommendation"])
def test_missing_unit_conflict_and_unsupported_ranking_cannot_pass(reason):
    candidate = _candidate("对象甲的数值为60，适用范围尚未明确。")
    review = parse_evidence_scope_review(_review_payload([candidate], reason), [candidate])
    assert not review.verified
    assert "60" in render_unverified_evidence(review)


@pytest.mark.parametrize("mutation", ["out_of_range", "invented_id", "reversed_range", "noninteger_range", "missing_support", "contradictory_status", "duplicate_field", "unknown_reason"])
def test_review_cannot_invent_quote_scope_or_bypass_status(mutation):
    candidates = [_candidate("当前数值60。"), _candidate("历史数量12。", chunk=1)]
    payload = json.loads(_review_payload(candidates, "relation"))
    if mutation == "out_of_range":
        payload["support"][0]["line_end"] = 2
    elif mutation == "invented_id":
        payload["support"][0]["source_id"] = "invented"
    elif mutation == "reversed_range":
        payload["support"][0]["line_start"] = 2
    elif mutation == "noninteger_range":
        payload["support"][0]["line_start"] = True
    elif mutation == "missing_support":
        payload["support"] = []
        payload["issues"][0]["evidence"] = []
    elif mutation == "contradictory_status":
        payload["status"] = "VERIFIED"
    elif mutation == "unknown_reason":
        payload["issues"][0]["reason"] = "business_specific_rule"
    raw = json.dumps(payload, ensure_ascii=False)
    if mutation == "duplicate_field":
        raw = raw[:-1] + ',"status":"VERIFIED"}'
    review = parse_evidence_scope_review(raw, candidates)
    assert review.error == "invalid_review"
    assert not review.verified


def test_review_prompt_contains_exact_context_and_all_generic_guards():
    candidates = [_candidate("当前对象甲：数值60。\n历史对象乙：数量12。")]
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(text=_review_payload(candidates, "relation"))

    review = review_generated_evidence_scope(
        answer_text="假设二者对应则可计算，但不能据此排名。", question="比较这些对象",
        source_candidates=candidates,
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)), model_id="offline",
    )
    assert not review.verified and len(calls) == 1
    prompt = calls[0]["contents"]
    assert candidates[0].source_id in prompt
    assert "历史对象乙" in prompt
    assert "条件性推演" in prompt and "跨chunk" in prompt
    assert "support" in calls[0]["config"]["response_schema"]["required"]


@pytest.mark.parametrize("domain,text,bad_fact,reason", CASES)
def test_runner_rejects_all_draft_fields_before_state_and_selection(
    monkeypatch, tmp_path, domain, text, bad_fact, reason,
):
    from app import chat_loop as runtime

    paths = ["材料甲.md", "材料乙.md"]
    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        if "你是事实关系审核器" in kwargs["contents"]:
            data = json.loads(kwargs["contents"].split("只返回符合schema的JSON。\n", 1)[1])
            refs = [{"source_id": e["source_id"], "line_start": 1, "line_end": len(e["lines"])} for e in data["evidence"] if e["path"] == paths[0]]
            return SimpleNamespace(text=json.dumps({
                "status": "UNVERIFIED", "bindings": [], "issues": [{"reason": reason, "evidence": refs}], "support": refs,
            }, ensure_ascii=False))
        return SimpleNamespace(text=(
            f"推荐结论：甲最优，{bad_fact}。\n推荐对象：甲\n推荐理由：{bad_fact}。\n"
            f"横向比较：甲{bad_fact}，依据【材料甲.md】；乙资料待确认，依据【材料乙.md】。\n"
            f"下一步行动：马上选择甲，{bad_fact}。\n推荐对象来源：材料甲.md\n来源文件：材料甲.md、材料乙.md"
        ))

    _run_turns(
        monkeypatch, tmp_path, questions=[f"比较这几个{domain}并推荐一个。"],
        repo_paths=paths, repo_chunks=[text, "其他对象的信息待确认。"], state=ConversationState(),
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
        evidence_reviewer=review_generated_evidence_scope,
    )
    state = runtime.conversation_state
    assert len(calls) == 2  # one draft, one review; never a repair generation
    assert bad_fact not in state.last_answer_text
    assert "马上选择甲" not in state.last_answer_text
    assert "暂缓据此选择" in state.last_answer_text
    assert state.last_selected_candidate is None and not state.last_selected_source_files
    assert state.last_answer_source_files == [paths[0]]
    assert all(line in state.last_answer_text for line in text.splitlines())


def test_unavailable_review_preserves_previous_answer(monkeypatch, tmp_path):
    from app import chat_loop as runtime

    calls = []

    def generate_content(**kwargs):
        calls.append(kwargs)
        if "你是事实关系审核器" in kwargs["contents"]:
            raise RuntimeError("offline")
        return SimpleNamespace(text="横向比较：甲数值60，来源【材料甲.md】；乙数值90，来源【材料乙.md】。")

    _run_turns(
        monkeypatch, tmp_path, questions=["比较这些方案并推荐一个。"],
        repo_paths=["材料甲.md", "材料乙.md"], repo_chunks=["甲数值60。", "乙数值90。"],
        state=ConversationState(last_answer_text="上一轮有效回答"),
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
        evidence_reviewer=review_generated_evidence_scope,
    )
    assert runtime.conversation_state.last_answer_text == "上一轮有效回答"
    assert len(calls) == 2


def test_rollback_is_explicit_and_does_not_call_remote(monkeypatch):
    monkeypatch.setenv("DOCMIND_EVIDENCE_SCOPE_BINDING", "0")
    assert build_evidence_scope_contract() == ""
    assert review_generated_evidence_scope(
        answer_text="原回答", question="原问题", source_candidates=[], client=None, model_id="offline",
    ).verified


@pytest.mark.parametrize("basis,assertion,expected", [
    ("COOCCURRENCE", "CONFIRMED", False), ("MISSING", "CONFIRMED", False),
    ("CONFLICT", "CONFIRMED", False), ("CONDITIONAL", "CONFIRMED", False),
    ("MISSING", "UNKNOWN", True), ("MISSING", "CONDITIONAL", True),
    ("DERIVED", "CONFIRMED", True), ("CONDITIONAL", "CONDITIONAL", True),
])
def test_binding_status_overrides_verdict_without_rejecting_honest_unknowns(basis, assertion, expected):
    candidate = _candidate("套餐甲：总价1200元，服务期12个月。")
    payload = json.loads(_review_payload([candidate]))
    payload["bindings"] = [{"claim": "用于审核的关系", "basis": basis, "assertion": assertion, "evidence": [_ref(candidate)]}]
    review = parse_evidence_scope_review(json.dumps(payload), [candidate])
    assert not review.error
    assert review.verified is expected


def test_contextless_followup_never_calls_generation_or_review(monkeypatch, tmp_path):
    def forbidden(**kwargs):
        pytest.fail("Contextless followup must not call any remote model")

    _run_turns(
        monkeypatch, tmp_path, questions=["比较这些商品，推荐一个。"],
        repo_paths=["合成资料甲.md", "合成资料乙.md"], state=ConversationState(),
        client=SimpleNamespace(models=SimpleNamespace(generate_content=forbidden)), evidence_reviewer=forbidden,
    )


def test_verified_derivation_preserves_selection_and_exact_followup_scope(monkeypatch, tmp_path):
    from app import chat_loop as runtime

    calls = []
    review_paths = []
    paths = ["材料甲.md", "材料乙.md"]

    def generate_content(**kwargs):
        calls.append(kwargs)
        if "你是事实关系审核器" in kwargs["contents"]:
            data = json.loads(kwargs["contents"].split("只返回符合schema的JSON。\n", 1)[1])
            review_paths.append([e["path"] for e in data["evidence"]])
            refs = [{"source_id": e["source_id"], "line_start": 1, "line_end": len(e["lines"])} for e in data["evidence"]]
            return SimpleNamespace(text=json.dumps({
                "support": refs, "bindings": [{"claim": "甲每支5元", "assertion": "CONFIRMED", "basis": "DERIVED", "evidence": refs}],
                "issues": [], "status": "VERIFIED",
            }))
        if len(calls) > 2:
            return SimpleNamespace(text="合成方案甲每支5元，依据【材料甲.md】的12支装60元。")
        return SimpleNamespace(text=(
            "推荐结论：就每支价格而言甲更低。\n推荐对象：合成方案甲\n推荐理由：甲每支5元。\n"
            "横向比较：甲12支装60元，平均5元/支，依据【材料甲.md】；乙每支8元，依据【材料乙.md】。\n"
            "推荐对象来源：材料甲.md\n来源文件：材料甲.md、材料乙.md"
        ))

    _run_turns(
        monkeypatch, tmp_path, questions=["比较这几个方案并推荐一个。", "详细分析一下"],
        repo_paths=paths, repo_chunks=["合成方案甲：12支装，售价60元。", "合成方案乙：每支8元。"],
        state=ConversationState(), client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
        evidence_reviewer=review_generated_evidence_scope,
    )
    assert len(calls) == 4
    assert review_paths == [paths, [paths[0]]]
    assert runtime.conversation_state.last_selected_candidate == "合成方案甲"
    assert runtime.conversation_state.last_selected_source_files == [paths[0]]
    assert "5元" in runtime.conversation_state.last_answer_text


def test_same_chunk_quotes_keep_intervening_material_scope_and_time():
    candidate = _candidate("当前对象甲：价格60元。\n历史记录\n旧版本：\n数量12个。")
    payload = {
        "support": [{"source_id": candidate.source_id, "line_start": 1, "line_end": 1}],
        "bindings": [], "status": "UNVERIFIED",
        "issues": [{"reason": "relation", "evidence": [
            {"source_id": candidate.source_id, "line_start": 4, "line_end": 4},
        ]}],
    }
    review = parse_evidence_scope_review(json.dumps(payload), [candidate])
    assert len(review.evidence) == 1
    assert review.evidence[0].quote == candidate.text
    assert "历史记录\n旧版本" in render_unverified_evidence(review)
    assert not review.verified


def test_verified_verdict_without_any_binding_check_is_not_accepted():
    candidate = _candidate("当前数值60，范围未知。")
    payload = json.loads(_review_payload([candidate]))
    payload["bindings"] = []
    assert parse_evidence_scope_review(json.dumps(payload), [candidate]).error == "invalid_review"


def test_factual_rejection_clears_previous_selection_and_keeps_its_evidence(monkeypatch, tmp_path):
    from app import chat_loop as runtime

    def generate_content(**kwargs):
        if "你是事实关系审核器" in kwargs["contents"]:
            data = json.loads(kwargs["contents"].split("只返回符合schema的JSON。\n", 1)[1])
            refs = [{"source_id": e["source_id"], "line_start": 1, "line_end": len(e["lines"])} for e in data["evidence"]]
            return SimpleNamespace(text=json.dumps({
                "support": refs, "bindings": [], "issues": [{"reason": "relation", "evidence": refs}],
                "status": "UNVERIFIED",
            }))
        return SimpleNamespace(text="设备甲内存32GB，完全满足要求。来源【材料甲.md】。")

    _run_turns(
        monkeypatch, tmp_path, questions=["请分析设备甲的内存配置及其适用范围。"],
        repo_paths=["材料甲.md"], repo_chunks=["设备甲售价900元。\n设备乙内存32GB。"],
        state=ConversationState(last_selected_candidate="设备甲", last_selected_source_files=["材料甲.md"]),
        client=SimpleNamespace(models=SimpleNamespace(generate_content=generate_content)),
        evidence_reviewer=review_generated_evidence_scope,
    )
    assert runtime.conversation_state.last_selected_candidate is None
    assert runtime.conversation_state.last_selected_source_files is None
    assert runtime.conversation_state.last_answer_source_files == ["材料甲.md"]
    assert "设备甲内存32GB" not in runtime.conversation_state.last_answer_text
