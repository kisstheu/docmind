from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.chat_state_helpers import update_state_after_retrieval_answer
from app.dialog.question_scope import (
    analyze_question_signals,
    decide_file_result_set_scope,
)
from app.dialog.result_set import (
    GeneratedResultSetProvenance,
    materialize_structured_generated_result_set,
    render_structured_generated_result_set,
)
from app.dialog.state_machine import ConversationState


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
    assert not state.last_generated_result_focuses

    _signals, selection = _scope_facts("第1个怎么样？", state, "result_set_followup")
    assert selection.selected_file_paths is None
    assert selection.file_result_set_selection is not None
    assert "无法可靠确定" in (
        selection.file_result_set_selection.rejection or ""
    )


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
