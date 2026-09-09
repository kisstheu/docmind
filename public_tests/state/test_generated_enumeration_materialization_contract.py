from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.chat_state_helpers import (
    update_state_after_answer_presentation,
    update_state_after_retrieval_answer,
)
from app.dialog.question_scope import (
    analyze_question_signals,
    decide_file_result_set_scope,
)
from app.dialog.result_set import (
    GeneratedResultSetProvenance,
    materialize_structured_generated_result_set,
    render_structured_generated_result_set,
    structured_generated_enumeration_schema,
)
from app.dialog.state_machine import ConversationState
from app.dialog.state_machine import detect_dialog_event
from retrieval.search_context import build_context_source_candidates, build_context_text


class _LoggerStub:
    def debug(self, *_args, **_kwargs):
        return None

    info = debug
    warning = debug
    error = debug


def _source_file_state() -> ConversationState:
    paths = ["旧资料甲.md", "项目资料甲.md", "项目资料乙.md"]
    answer = "\n".join(f"{index}. {path}" for index, path in enumerate(paths, 1))
    return ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_content_user_question="有哪些文档？",
        last_effective_search_query="合成资料",
        last_answer_text=answer,
        last_answer_preview=answer,
        last_answer_type="enumeration_file",
        last_result_set_items=paths,
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
    )


def _repo_state():
    return SimpleNamespace(
        paths=["旧资料甲.md", "项目资料甲.md", "项目资料乙.md"],
        docs=[
            "这里只记录合成背景。",
            "标题：项目甲\n里程碑：完成合成方案评审。",
            "标题：项目乙\n里程碑：完成合成最终验收。",
        ],
    )


def _indexed_source_repo(documents: dict[str, str]):
    paths = list(documents)
    texts = list(documents.values())
    return SimpleNamespace(
        paths=paths,
        docs=texts,
        chunk_paths=paths,
        chunk_texts=texts,
        chunk_meta=[
            {"chunk_id": 0, "start": 0, "end": len(text)}
            for text in texts
        ],
    )


def _source_candidates(repo_state):
    return build_context_source_candidates(
        list(range(len(repo_state.chunk_texts))),
        repo_state,
        per_file_limit=1,
    )


def _payload():
    return {
        "items": [
            {
                "display_name": "项目甲",
                "source_paths": ["项目资料甲.md"],
                "evidence_text": "标题：项目甲",
            },
            {
                "display_name": "项目乙",
                "source_paths": ["项目资料乙.md"],
                "evidence_text": "标题：项目乙",
            },
        ]
    }


def _scope_facts(question: str, state: ConversationState, event_name: str):
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name=event_name,
    )
    return signals, scope


def _write_generated_result(
    state: ConversationState,
    provenance: GeneratedResultSetProvenance,
    *,
    answer: str | None = None,
) -> ConversationState:
    question = "有哪些项目？"
    signals, scope = _scope_facts(question, state, "result_set_followup")
    return update_state_after_retrieval_answer(
        state,
        question,
        answer if answer is not None else render_structured_generated_result_set(provenance),
        _LoggerStub(),
        event_name="result_set_followup",
        generated_result_provenance=provenance,
        question_signals=signals,
        scope_decision=scope,
    )


def test_reliable_generic_generated_enumeration_replaces_old_file_authority():
    provenance = materialize_structured_generated_result_set(
        _payload(),
        entity_type="项目",
        candidate_paths=_repo_state().paths,
        repo_state=_repo_state(),
    )
    state = _write_generated_result(_source_file_state(), provenance)

    assert provenance.reliable is True
    assert state.last_result_set_entity_type == "项目"
    assert state.last_result_set_items == ["项目甲", "项目乙"]
    assert state.last_result_set_selectable is True
    assert state.last_generated_result_source_hits == [
        ["项目资料甲.md"],
        ["项目资料乙.md"],
    ]
    assert len(state.last_generated_result_focuses or []) == 2

    signals, selection = _scope_facts("第1个怎么样？", state, "result_set_followup")

    assert signals.explicit_single_file_result_reference is True
    assert signals.document_evaluation_request is True
    assert selection.selected_file_paths == ("项目资料甲.md",)
    assert selection.selected_file_paths != ("旧资料甲.md",)
    assert selection.query_result_set_items == ("项目甲",)
    assert selection.file_result_set_selection.display_item == "项目甲"
    assert selection.file_result_set_selection.opaque_focus.startswith(
        "core-generated:v1:"
    )


def test_generic_generated_ordinal_keeps_detail_question_semantics():
    provenance = materialize_structured_generated_result_set(
        _payload(),
        entity_type="项目",
        candidate_paths=_repo_state().paths,
        repo_state=_repo_state(),
    )
    state = _write_generated_result(_source_file_state(), provenance)

    signals, selection = _scope_facts(
        "第1个里程碑是什么？",
        state,
        "result_set_followup",
    )

    assert signals.document_evaluation_request is False
    assert selection.selected_file_paths == ("项目资料甲.md",)
    assert selection.file_result_set_selection.display_item == "项目甲"


def test_generated_ordinal_does_not_claim_an_explicit_other_ordinal_target():
    provenance = materialize_structured_generated_result_set(
        _payload(),
        entity_type="项目",
        candidate_paths=_repo_state().paths,
        repo_state=_repo_state(),
    )
    state = _write_generated_result(_source_file_state(), provenance)

    _signals, selection = _scope_facts(
        "第1个问题是什么？",
        state,
        "result_set_followup",
    )

    assert selection.selected_file_paths is None


def test_generated_ordinal_detail_question_is_domain_neutral():
    path = "合同资料.md"
    document = "标题：合成合同甲\n交付条件：完成合成验收。"
    provenance = materialize_structured_generated_result_set(
        {
            "items": [
                {
                    "display_name": "合成合同甲",
                    "source_paths": [path],
                    "evidence_text": "标题：合成合同甲",
                }
            ]
        },
        entity_type="合同",
        candidate_paths=[path],
        repo_state=SimpleNamespace(paths=[path], docs=[document]),
    )
    state = _write_generated_result(_source_file_state(), provenance)

    signals, selection = _scope_facts(
        "第1个交付条件是什么？",
        state,
        "result_set_followup",
    )

    assert signals.document_evaluation_request is False
    assert selection.selected_file_paths == (path,)
    assert selection.file_result_set_selection.display_item == "合成合同甲"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "items": [
                {
                    "display_name": "项目甲",
                    "source_paths": [],
                    "evidence_text": "",
                }
            ]
        },
        {
            "items": [
                _payload()["items"][0],
                {
                    "display_name": "项目乙",
                    "source_paths": ["项目资料乙.md"],
                    "evidence_text": "不存在的证据",
                },
            ]
        },
        {"items": []},
        None,
    ],
    ids=["missing-binding", "partial-binding", "empty", "parse-failure"],
)
def test_unreliable_structured_enumeration_keeps_old_context_but_disables_ordinal(
    payload,
):
    provenance = materialize_structured_generated_result_set(
        payload,
        entity_type="项目",
        candidate_paths=_repo_state().paths,
        repo_state=_repo_state(),
    )
    state = _write_generated_result(_source_file_state(), provenance)

    assert provenance.reliable is False
    assert state.last_result_set_items == _repo_state().paths
    assert state.last_result_set_entity_type == "文件"
    assert state.last_result_set_selectable is False
    assert state.last_generated_result_entity_type == (
        "项目" if provenance.display_items else None
    )
    assert not state.last_generated_result_focuses

    _signals, selection = _scope_facts("第1个怎么样？", state, "result_set_followup")
    assert selection.selected_file_paths is None
    assert selection.file_result_set_selection is not None
    assert "无法可靠确定" in (
        selection.file_result_set_selection.rejection or ""
    )


@pytest.mark.parametrize(
    ("entity_type", "items", "question"),
    [
        ("岗位", ["合成岗位甲", "合成岗位乙"], "这些有学历要求吗？"),
        ("合同事项", ["合成交付事项", "合成验收事项"], "有时间限制吗？"),
        ("采购项", ["合成设备甲", "合成服务乙"], "有价格区间吗？"),
        ("课程", ["合成课程甲", "合成课程乙"], "这些有先修条件吗？"),
    ],
)
def test_unbound_answer_entities_are_semantic_scope_not_selectable_result_set(
    entity_type,
    items,
    question,
):
    state = _source_file_state()
    state.last_route = "normal_retrieval"
    state.last_content_route = "normal_retrieval"
    state.last_content_user_question = f"有哪些{entity_type}？"
    state.last_effective_search_query = entity_type
    state.last_answer_text = "\n".join(
        f"{index}. {item}" for index, item in enumerate(items, 1)
    )
    state.last_answer_preview = state.last_answer_text
    state.last_answer_type = None
    state.last_result_set_selectable = False
    state.last_generated_result_items = list(items)
    state.last_generated_result_entity_type = entity_type
    state.last_generated_result_source_candidates = list(_repo_state().paths)
    state.last_generated_result_source_hits = [[] for _item in items]

    event = detect_dialog_event(question, state, _LoggerStub())
    signals, scope = _scope_facts(question, state, event.name)

    assert event.name == "content_followup"
    assert scope.answer_entity_followup is True
    assert scope.result_scope_paths == tuple(_repo_state().paths)
    assert scope.query_result_set_items == tuple(items)
    assert scope.query_result_set_entity == entity_type
    assert state.last_result_set_items == _repo_state().paths
    assert state.last_result_set_entity_type == "文件"
    assert state.last_result_set_selectable is False

    _signals, ordinal_scope = _scope_facts("第1个怎么样？", state, "content_followup")
    assert ordinal_scope.answer_entity_followup is False
    assert ordinal_scope.selected_file_paths is None
    assert ordinal_scope.file_result_set_selection is not None
    assert ordinal_scope.file_result_set_selection.rejection is not None


def test_standalone_question_does_not_reuse_unbound_answer_entity_scope():
    state = _source_file_state()
    state.last_generated_result_items = ["合成岗位甲", "合成岗位乙"]
    state.last_generated_result_entity_type = "岗位"
    state.last_generated_result_source_candidates = list(_repo_state().paths)
    state.last_generated_result_source_hits = [[], []]

    question = "学历认证是什么意思？"
    event = detect_dialog_event(question, state, _LoggerStub())
    signals, scope = _scope_facts(question, state, event.name)

    assert signals.standalone_general_question is True
    assert scope.answer_entity_followup is False
    assert scope.query_result_set_items is None


def test_complete_collection_summary_enumeration_property_followup_state_chain():
    files = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    state = ConversationState()
    list_answer = "当前知识库里的文件如下：\n" + "\n".join(
        f"{index}. {path}" for index, path in enumerate(files, 1)
    )
    from app.chat_state_helpers import update_state_after_local_answer

    state = update_state_after_local_answer(
        state,
        question="有哪些文档？",
        answer=list_answer,
        route="repo_meta",
        local_topic="list_files",
        is_content_answer=True,
    )
    assert state.last_result_set_items == files

    summary_question = "是关于啥的？"
    summary_signals, summary_scope = _scope_facts(
        summary_question,
        state,
        "result_set_followup",
    )
    state = update_state_after_retrieval_answer(
        state,
        summary_question,
        "这些合成材料共同描述项目交付与验收。",
        _LoggerStub(),
        event_name="result_set_followup",
        question_signals=summary_signals,
        scope_decision=summary_scope,
    )
    assert state.last_result_set_items == files

    repo_state = SimpleNamespace(
        paths=files,
        docs=[
            "标题：合成事项甲\n期限：三十日。",
            "标题：合成事项乙\n期限：十日。",
            "这里只记录合成背景。",
        ],
    )
    provenance = materialize_structured_generated_result_set(
        {
            "items": [
                {
                    "display_name": "合成事项甲",
                    "source_paths": [files[0]],
                    "evidence_text": "非原文甲",
                },
                {
                    "display_name": "合成事项乙",
                    "source_paths": [files[1]],
                    "evidence_text": "非原文乙",
                },
            ]
        },
        entity_type="事项",
        candidate_paths=files,
        repo_state=repo_state,
    )
    enumeration_question = "总结一下有哪些事项？"
    enumeration_signals, enumeration_scope = _scope_facts(
        enumeration_question,
        state,
        "synthesis_request",
    )
    state = update_state_after_retrieval_answer(
        state,
        enumeration_question,
        render_structured_generated_result_set(provenance),
        _LoggerStub(),
        event_name="synthesis_request",
        generated_result_provenance=provenance,
        question_signals=enumeration_signals,
        scope_decision=enumeration_scope,
    )

    assert provenance.source_hits == ((), ())
    assert state.last_result_set_items == files
    assert state.last_result_set_entity_type == "文件"
    assert state.last_result_set_selectable is False
    assert state.last_generated_result_items == ["合成事项甲", "合成事项乙"]
    assert state.last_generated_result_entity_type == "事项"

    property_question = "有时间限制吗？"
    event = detect_dialog_event(property_question, state, _LoggerStub())
    _signals, property_scope = _scope_facts(
        property_question,
        state,
        event.name,
    )
    assert event.name == "content_followup"
    assert property_scope.answer_entity_followup is True
    assert property_scope.result_scope_paths == tuple(files)
    assert property_scope.query_result_set_items == ("合成事项甲", "合成事项乙")
    assert property_scope.query_result_set_entity == "事项"

    property_answer = "合成事项甲为三十日，合成事项乙为十日。"
    property_signals, property_scope = _scope_facts(
        property_question,
        state,
        event.name,
    )
    state = update_state_after_retrieval_answer(
        state,
        property_question,
        property_answer,
        _LoggerStub(),
        event_name=event.name,
        question_signals=property_signals,
        scope_decision=property_scope,
    )

    presentation_question = "给我个表格吧"
    presentation_event = detect_dialog_event(
        presentation_question,
        state,
        _LoggerStub(),
    )
    presentation_signals, presentation_scope = _scope_facts(
        presentation_question,
        state,
        presentation_event.name,
    )
    assert presentation_event.name == "answer_presentation_followup"
    assert presentation_scope.answer_entity_followup is False
    assert presentation_scope.result_scope_paths is None
    assert presentation_scope.requires_result_set_generation is False
    assert state.last_answer_text == property_answer

    state = update_state_after_answer_presentation(
        state,
        presentation_question,
        "| 事项 | 期限 |\n| --- | --- |\n| 合成事项甲 | 三十日 |",
    )
    next_question = "都有验收条件吗？"
    next_event = detect_dialog_event(next_question, state, _LoggerStub())
    _signals, next_scope = _scope_facts(next_question, state, next_event.name)
    assert next_event.name == "content_followup"
    assert next_scope.answer_entity_followup is True
    assert next_scope.result_scope_paths == tuple(files)
    assert next_scope.query_result_set_items == ("合成事项甲", "合成事项乙")
    assert next_scope.query_result_set_entity == "事项"


def test_display_and_structured_item_count_mismatch_fails_closed():
    provenance = materialize_structured_generated_result_set(
        _payload(),
        entity_type="项目",
        candidate_paths=_repo_state().paths,
        repo_state=_repo_state(),
    )
    state = _write_generated_result(
        _source_file_state(),
        provenance,
        answer="1. 项目甲\n   来源文件：项目资料甲.md",
    )

    assert provenance.reliable is True
    assert state.last_result_set_items == _repo_state().paths
    assert state.last_result_set_selectable is False
    assert not state.last_generated_result_focuses


def test_materialized_generated_ordinal_bounds_and_illegal_form_do_not_select():
    provenance = materialize_structured_generated_result_set(
        _payload(),
        entity_type="项目",
        candidate_paths=_repo_state().paths,
        repo_state=_repo_state(),
    )
    state = _write_generated_result(_source_file_state(), provenance)

    _signals, overflow = _scope_facts("第3个怎么样？", state, "result_set_followup")
    _signals, illegal = _scope_facts("第0个怎么样？", state, "result_set_followup")

    assert overflow.selected_file_paths is None
    assert "只有 2 个条目" in (overflow.file_result_set_selection.rejection or "")
    assert illegal.selected_file_paths is None


def test_source_outside_bounded_collection_cannot_become_authority():
    payload = _payload()
    payload["items"][0]["source_paths"] = ["项目资料甲.md"]
    provenance = materialize_structured_generated_result_set(
        payload,
        entity_type="项目",
        candidate_paths=["旧资料甲.md", "项目资料乙.md"],
        repo_state=_repo_state(),
    )

    assert provenance.reliable is False
    assert provenance.opaque_focuses == ()
    assert provenance.binding_failures[0] == "source_outside_scope"


@pytest.mark.parametrize(
    ("evidence_text", "expected_failure"),
    [
        ("这里没有这段原文", "evidence_not_exact"),
        ("里程碑：完成合成方案评审。", "display_not_in_evidence"),
    ],
)
def test_structured_binding_reports_domain_neutral_item_failure(
    evidence_text,
    expected_failure,
):
    provenance = materialize_structured_generated_result_set(
        {
            "items": [
                {
                    "display_name": "项目甲",
                    "source_paths": ["项目资料甲.md"],
                    "evidence_text": evidence_text,
                }
            ]
        },
        entity_type="项目",
        candidate_paths=_repo_state().paths,
        repo_state=_repo_state(),
    )

    assert provenance.reliable is False
    assert provenance.source_hits == ((),)
    assert provenance.binding_failures == (expected_failure,)


@pytest.mark.parametrize(
    ("entity_type", "display_name", "path", "document"),
    [
        ("项目", "合成项目甲", "项目资料.md", "名称：合成项目甲\n阶段：方案评审"),
        ("课程", "合成课程乙", "课程资料.md", "标题：合成课程乙\n课时：八课时"),
        ("产品", "合成产品丙", "产品资料.md", "产品名：合成产品丙\n状态：验证中"),
    ],
)
def test_structured_materialization_is_domain_neutral(
    entity_type,
    display_name,
    path,
    document,
):
    provenance = materialize_structured_generated_result_set(
        {
            "items": [
                {
                    "display_name": display_name,
                    "source_paths": [path],
                    "evidence_text": document.splitlines()[0],
                }
            ]
        },
        entity_type=entity_type,
        candidate_paths=[path],
        repo_state=SimpleNamespace(paths=[path], docs=[document]),
    )

    assert provenance.reliable is True
    assert provenance.entity_type == entity_type
    assert provenance.display_items == (display_name,)
    assert provenance.source_hits == ((path,),)
    assert provenance.opaque_focuses[0].startswith("core-generated:v1:")


def test_canonical_source_identity_binds_multiple_grounded_items():
    repo_state = _indexed_source_repo(
        {
            "资料A.md": "名称：Alpha\n说明：第一项。",
            "资料B.md": "名称：Beta\n说明：第二项。",
            "资料C.md": "名称：Gamma\n说明：第三项。",
        }
    )
    candidates = _source_candidates(repo_state)
    source_ids = {candidate.path: candidate.source_id for candidate in candidates}

    provenance = materialize_structured_generated_result_set(
        {
            "items": [
                {
                    "display_name": "Alpha",
                    "source_ids": [source_ids["资料A.md"]],
                    "evidence_text": "名称：Alpha",
                },
                {
                    "display_name": "Beta",
                    "source_ids": [source_ids["资料B.md"]],
                    "evidence_text": "名称：Beta",
                },
                {
                    "display_name": "Gamma",
                    "source_ids": [source_ids["资料C.md"]],
                    "evidence_text": "名称：Gamma",
                },
            ]
        },
        entity_type="条目",
        candidate_paths=repo_state.paths,
        repo_state=repo_state,
        source_candidates=candidates,
    )

    assert provenance.reliable is True
    assert provenance.source_hits == (
        ("资料A.md",),
        ("资料B.md",),
        ("资料C.md",),
    )
    assert provenance.source_id_hits == (
        (source_ids["资料A.md"],),
        (source_ids["资料B.md"],),
        (source_ids["资料C.md"],),
    )
    assert provenance.evidence_hits == tuple(
        (document,) for document in repo_state.chunk_texts
    )

    schema = structured_generated_enumeration_schema(
        tuple(source_ids.values())
    )
    source_id_items = schema["properties"]["items"]["items"]["properties"][
        "source_ids"
    ]["items"]
    assert source_id_items["enum"] == list(source_ids.values())


def test_canonical_source_identity_is_stable_when_candidate_order_changes():
    repo_state = _indexed_source_repo(
        {
            "资料A.md": "名称：Alpha",
            "资料B.md": "名称：Beta",
        }
    )
    candidates = _source_candidates(repo_state)
    alpha_id = next(
        candidate.source_id for candidate in candidates if candidate.path == "资料A.md"
    )

    provenance = materialize_structured_generated_result_set(
        {
            "items": [
                {
                    "display_name": "Alpha",
                    "source_ids": [alpha_id],
                    "evidence_text": "名称：Alpha",
                }
            ]
        },
        entity_type="条目",
        candidate_paths=list(reversed(repo_state.paths)),
        repo_state=repo_state,
        source_candidates=tuple(reversed(candidates)),
    )

    assert provenance.reliable is True
    assert provenance.source_hits == (("资料A.md",),)
    assert provenance.source_id_hits == ((alpha_id,),)


def test_similar_source_texts_are_distinguished_by_canonical_identity():
    repo_state = _indexed_source_repo(
        {
            "资料A.md": "名称：Alpha\n阶段：初版。",
            "资料B.md": "名称：Alpha\n阶段：扩展版。",
        }
    )
    candidates = _source_candidates(repo_state)
    source_ids = {candidate.path: candidate.source_id for candidate in candidates}

    provenance = materialize_structured_generated_result_set(
        {
            "items": [
                {
                    "display_name": "Alpha",
                    "source_ids": [source_ids["资料B.md"]],
                    "evidence_text": "名称：Alpha；阶段：扩展版。",
                }
            ]
        },
        entity_type="条目",
        candidate_paths=repo_state.paths,
        repo_state=repo_state,
        source_candidates=candidates,
    )

    assert provenance.reliable is True
    assert provenance.source_hits == (("资料B.md",),)
    assert provenance.source_id_hits == ((source_ids["资料B.md"],),)


def test_evidence_formatting_change_does_not_replace_canonical_provenance():
    repo_state = _indexed_source_repo(
        {"资料A.md": "名称：Alpha（稳定版本）\n状态：已验证。"}
    )
    candidates = _source_candidates(repo_state)

    provenance = materialize_structured_generated_result_set(
        {
            "items": [
                {
                    "display_name": "Alpha",
                    "source_ids": [candidates[0].source_id],
                    "evidence_text": "名称: **Alpha** (稳定版本)",
                }
            ]
        },
        entity_type="条目",
        candidate_paths=repo_state.paths,
        repo_state=repo_state,
        source_candidates=candidates,
    )

    assert "名称: **Alpha** (稳定版本)" not in repo_state.docs[0]
    assert provenance.reliable is True
    assert provenance.binding_failures == ("",)


@pytest.mark.parametrize(
    ("source_id", "display_name", "evidence_text", "expected_failure"),
    [
        ("retrieval-source:v1:not-present", "Alpha", "名称：Alpha", "source_id_not_found"),
        ("gamma", "Alpha", "名称：Alpha", "display_not_in_source"),
        ("alpha", "Delta", "名称：Delta", "display_not_in_source"),
        ("alpha", "Alpha", "这里只是格式化证据", "display_not_in_evidence"),
    ],
    ids=[
        "unknown-source-id",
        "source-id-evidence-mismatch",
        "unsupported-item",
        "display-not-in-evidence",
    ],
)
def test_canonical_source_binding_rejects_untrusted_or_unsupported_items(
    source_id,
    display_name,
    evidence_text,
    expected_failure,
):
    repo_state = _indexed_source_repo(
        {
            "资料A.md": "名称：Alpha",
            "资料C.md": "名称：Gamma",
        }
    )
    candidates = _source_candidates(repo_state)
    candidate_ids = {
        "alpha": candidates[0].source_id,
        "gamma": candidates[1].source_id,
    }

    provenance = materialize_structured_generated_result_set(
        {
            "items": [
                {
                    "display_name": display_name,
                    "source_ids": [candidate_ids.get(source_id, source_id)],
                    "evidence_text": evidence_text,
                }
            ]
        },
        entity_type="条目",
        candidate_paths=repo_state.paths,
        repo_state=repo_state,
        source_candidates=candidates,
    )

    assert provenance.reliable is False
    assert provenance.source_hits == ((),)
    assert provenance.source_id_hits == ((),)
    assert provenance.evidence_hits == ((),)
    assert provenance.binding_failures == (expected_failure,)


def test_canonical_binding_keeps_full_set_fail_closed_for_mixed_output():
    repo_state = _indexed_source_repo(
        {
            "资料A.md": "名称：Alpha",
            "资料B.md": "名称：Beta",
        }
    )
    candidates = _source_candidates(repo_state)

    provenance = materialize_structured_generated_result_set(
        {
            "items": [
                {
                    "display_name": "Alpha",
                    "source_ids": [candidates[0].source_id],
                    "evidence_text": "名称：Alpha",
                },
                {
                    "display_name": "Delta",
                    "source_ids": [candidates[1].source_id],
                    "evidence_text": "名称：Delta",
                },
            ]
        },
        entity_type="条目",
        candidate_paths=repo_state.paths,
        repo_state=repo_state,
        source_candidates=candidates,
    )

    assert provenance.reliable is False
    assert provenance.source_hits == (("资料A.md",), ())
    assert provenance.evidence_hits == (("名称：Alpha",), ())
    assert provenance.opaque_focuses == ()
    assert render_structured_generated_result_set(provenance) == (
        "本轮未生成可可靠引用的枚举结果，请重试。"
    )


def test_duplicate_item_can_bind_to_multiple_canonical_sources_once():
    repo_state = _indexed_source_repo(
        {
            "资料A.md": "名称：Alpha\n说明：初版。",
            "资料B.md": "名称：Alpha\n说明：修订版。",
        }
    )
    candidates = _source_candidates(repo_state)

    provenance = materialize_structured_generated_result_set(
        {
            "items": [
                {
                    "display_name": "Alpha",
                    "source_ids": [candidate.source_id for candidate in candidates],
                    "evidence_text": "名称：Alpha",
                }
            ]
        },
        entity_type="条目",
        candidate_paths=repo_state.paths,
        repo_state=repo_state,
        source_candidates=candidates,
    )

    assert provenance.reliable is True
    assert provenance.display_items == ("Alpha",)
    assert provenance.source_hits == (("资料A.md", "资料B.md"),)
    assert provenance.source_id_hits == (
        tuple(candidate.source_id for candidate in candidates),
    )


def test_system_source_candidate_identity_is_recomputed_before_binding():
    repo_state = _indexed_source_repo({"资料A.md": "名称：Alpha"})
    candidate = _source_candidates(repo_state)[0]
    tampered_candidate = replace(
        candidate,
        source_id="retrieval-source:v1:tampered",
    )

    provenance = materialize_structured_generated_result_set(
        {
            "items": [
                {
                    "display_name": "Alpha",
                    "source_ids": [tampered_candidate.source_id],
                    "evidence_text": "名称：Alpha",
                }
            ]
        },
        entity_type="条目",
        candidate_paths=repo_state.paths,
        repo_state=repo_state,
        source_candidates=(tampered_candidate,),
    )

    assert provenance.reliable is False
    assert provenance.failure_reason == "invalid_source_candidates"


def test_canonical_source_ids_are_only_added_to_structured_generation_context():
    repo_state = _indexed_source_repo({"资料A.md": "名称：Alpha"})

    normal_context = build_context_text([0], repo_state, _LoggerStub())
    structured_context = build_context_text(
        [0],
        repo_state,
        _LoggerStub(),
        include_source_ids=True,
    )

    assert "证据【retrieval-source:v1:" not in normal_context
    assert "证据【retrieval-source:v1:" in structured_context
    assert "文件【资料A.md】" in normal_context
    assert "文件【资料A.md】" in structured_context
