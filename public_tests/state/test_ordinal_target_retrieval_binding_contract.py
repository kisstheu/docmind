from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from ai.table_presentation import StructuredTable
from ai.table_presentation import refine_structured_table
from app.chat_state_helpers import update_state_after_answer_presentation
from app.dialog.question_scope import analyze_question_signals, decide_file_result_set_scope
from app.dialog.result_set import (
    has_explicit_single_file_result_reference,
    materialize_single_file_result_set_question,
)
from app.dialog.state_machine import ConversationState, detect_dialog_event
from app.retrieval_flow.materials import build_retrieval_materials


class _LoggerStub:
    def debug(self, *_args, **_kwargs):
        return None

    info = debug
    warning = debug


class _EmbeddingStub:
    def encode(self, _texts):
        return np.asarray([[1.0, 0.0]], dtype=float)


def _state_with_reordered_presentation(paths: list[str]) -> ConversationState:
    original_answer = "\n".join(
        f"{index}. {path}" for index, path in enumerate(paths, 1)
    )
    return ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_content_user_question="这些材料的状态如何？",
        last_effective_search_query="合成材料状态",
        last_answer_text=original_answer,
        last_answer_preview=original_answer,
        last_factual_answer_text=original_answer,
        current_presentation_table=StructuredTable(
            columns=("文件", "状态"),
            rows=(
                (paths[2], "已确认"),
                (paths[0], "待确认"),
                (paths[1], "已确认"),
            ),
            row_identities=(paths[2], paths[0], paths[1]),
        ),
        last_answer_type="enumeration_file",
        last_result_set_items=list(paths),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
        last_result_set_focus_file=paths[2],
    )


def _scope(question: str, state: ConversationState, *, focus: str | None = None):
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    event = detect_dialog_event(question, state, _LoggerStub(), focused_file=focus)
    decision = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=focus,
        event_name=event.name,
    )
    return signals, event, decision


@pytest.mark.parametrize(
    ("question", "expected_index"),
    [
        ("那第一个描述了啥？", 2),
        ("第二份到底算有效的还是不算？", 0),
        ("第二篇是否已经确认？", 0),
        ("第二项的状态如何？", 0),
    ],
)
def test_natural_bare_ordinal_classifier_resolves_active_result_item(
    question,
    expected_index,
):
    paths = ["合成条目甲.md", "合成条目乙.md", "合成条目丙.md"]
    signals, event, decision = _scope(
        question,
        _state_with_reordered_presentation(paths),
        focus=paths[2],
    )

    assert signals.explicit_single_file_result_reference is True
    assert event.name == "result_set_followup"
    assert decision.selected_file_paths == (paths[expected_index],)
    assert decision.result_scope_paths == (paths[expected_index],)
    assert decision.query_result_set_items == (paths[expected_index],)


@pytest.mark.parametrize(
    "question",
    [
        "第二个问题到底算解决了还是没解决？",
        "第二章是否已经确认？",
        "第二个步骤如何执行？",
    ],
)
def test_adjacent_non_file_ordinals_do_not_select_active_file(question):
    paths = ["合成条目甲.md", "合成条目乙.md", "合成条目丙.md"]
    state = _state_with_reordered_presentation(paths)
    state.last_result_set_focus_file = None
    _signals, _event, decision = _scope(
        question,
        state,
        focus=None,
    )

    assert has_explicit_single_file_result_reference(question) is False
    assert decision.selected_file_paths is None
    assert decision.result_scope_paths is None


def test_presentation_sort_uses_visible_order_without_reordering_underlying_set():
    paths = ["合成资料甲.md", "合成记录乙.md", "合成说明丙.md"]
    state = _state_with_reordered_presentation(paths)

    _signals, _event, decision = _scope(
        "第二份到底算有效的还是不算？",
        state,
        focus=paths[2],
    )

    assert state.current_presentation_table.rows[1][0] == paths[0]
    assert state.last_result_set_items[1] == paths[1]
    assert decision.result_scope_paths == (paths[0],)


def test_no_presentation_keeps_underlying_result_set_ordinal_authority():
    paths = ["合成资料甲.md", "合成记录乙.md", "合成说明丙.md"]
    state = _state_with_reordered_presentation(paths)
    state.current_presentation_table = None

    _signals, _event, decision = _scope("第二份怎么样？", state)

    assert decision.result_scope_paths == (paths[1],)


def test_unreordered_presentation_keeps_equivalent_second_item_authority():
    paths = ["合成资料甲.md", "合成记录乙.md", "合成说明丙.md"]
    state = _state_with_reordered_presentation(paths)
    state.current_presentation_table = StructuredTable(
        columns=("文件", "状态"),
        rows=tuple((path, "已确认") for path in paths),
        row_identities=tuple(paths),
    )

    _signals, _event, decision = _scope("第二份怎么样？", state)

    assert decision.result_scope_paths == (paths[1],)


@pytest.mark.parametrize(
    "row_identities",
    [
        None,
        ("合成资料甲.md", "合成记录乙.md"),
        ("合成资料甲.md", "合成记录乙.md", "过期资料.md"),
        ("合成资料甲.md", "合成资料甲.md", "合成说明丙.md"),
    ],
)
def test_unmappable_or_stale_presentation_ordinal_fails_closed(row_identities):
    paths = ["合成资料甲.md", "合成记录乙.md", "合成说明丙.md"]
    state = _state_with_reordered_presentation(paths)
    state.current_presentation_table = StructuredTable(
        columns=("文件",),
        rows=((paths[2],), (paths[0],), (paths[1],)),
        row_identities=row_identities,
    )

    _signals, _event, decision = _scope("第二份怎么样？", state)

    assert decision.selected_file_paths is None
    assert decision.result_scope_paths is None
    assert decision.file_result_set_selection is not None
    assert "无法可靠确定" in (decision.file_result_set_selection.rejection or "")


def test_duplicate_display_names_do_not_create_guessed_row_identity():
    paths = ["合成目录甲/共享资料.md", "合成目录乙/共享资料.md"]
    state = ConversationState(
        last_answer_text="1. 共享资料.md\n2. 共享资料.md",
        last_answer_preview="1. 共享资料.md\n2. 共享资料.md",
        last_result_set_items=list(paths),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
    )
    table = StructuredTable(
        columns=("文件", "状态"),
        rows=(("共享资料.md", "甲"), ("共享资料.md", "乙")),
    )

    update_state_after_answer_presentation(
        state,
        "整理成表格",
        "本地渲染表格",
        table=table,
    )
    _signals, _event, decision = _scope("第一份怎么样？", state)

    assert state.current_presentation_table.row_identities is None
    assert decision.result_scope_paths is None
    assert "无法可靠确定" in (decision.file_result_set_selection.rejection or "")


def test_bound_row_identity_moves_with_local_sort_without_reordering_result_set():
    paths = ["合成资料甲.md", "合成记录乙.md", "合成说明丙.md"]
    state = _state_with_reordered_presentation(paths)
    state.current_presentation_table = None
    source_table = StructuredTable(
        columns=("文件", "优先级"),
        rows=((paths[0], "2"), (paths[1], "3"), (paths[2], "1")),
    )
    update_state_after_answer_presentation(
        state,
        "整理成表格",
        "本地渲染表格",
        table=source_table,
    )

    refinement = refine_structured_table(
        state.current_presentation_table,
        "按优先级排列",
    )
    update_state_after_answer_presentation(
        state,
        "按优先级排列",
        "本地排序表格",
        table=refinement.table,
        options=refinement.options,
    )

    assert refinement.valid is True
    assert state.current_presentation_table.row_identities == (
        paths[2],
        paths[0],
        paths[1],
    )
    assert state.last_result_set_items == paths


def test_resolved_ordinal_target_binds_retrieval_focus_evidence_and_question():
    paths = ["合成采购甲.md", "合成合同乙.md", "合成课程丙.md"]
    question = "那第一个包含了啥？"
    state = _state_with_reordered_presentation(paths)
    _signals, event, decision = _scope(question, state, focus=paths[2])
    resolved_target = decision.result_scope_paths[0]
    now = datetime.now()
    repo_state = SimpleNamespace(
        paths=paths,
        docs=["甲状态待确认", "乙状态已经确认", "丙状态待确认"],
        chunk_paths=paths,
        chunk_texts=["甲状态待确认", "乙状态已经确认", "丙状态待确认"],
        chunk_file_times=[now, now, now],
        chunk_embeddings=np.asarray(
            [[1.0, 0.0], [0.9, 0.0], [-1.0, 0.0]],
            dtype=float,
        ),
        chunk_meta=[
            {"chunk_id": 0, "start": 0, "end": 6},
            {"chunk_id": 0, "start": 0, "end": 8},
            {"chunk_id": 0, "start": 0, "end": 6},
        ],
    )

    materials = build_retrieval_materials(
        question=question,
        search_query=question,
        context_anchor="",
        flags={
            "skip_retrieval": False,
            "is_inventory_query": False,
            "inventory_target_type": None,
        },
        repo_state=repo_state,
        model_emb=_EmbeddingStub(),
        logger=_LoggerStub(),
        current_focus_file=resolved_target,
        event=event,
        allowed_paths=set(decision.result_scope_paths),
    )
    retrieved_paths = [
        repo_state.chunk_paths[index] for index in materials["relevant_indices"]
    ]
    materialized_question = materialize_single_file_result_set_question(
        question,
        resolved_target,
    )

    assert resolved_target == paths[2]
    assert materials["current_focus_file"] == resolved_target
    assert retrieved_paths == [resolved_target]
    assert resolved_target in materials["context_text"]
    assert paths[0] not in materials["context_text"]
    assert paths[1] not in materials["context_text"]
    assert materialized_question == f"文件《{resolved_target}》包含了啥？"


def test_scoped_ordinal_does_not_expand_when_target_has_no_indexed_evidence():
    paths = ["合成采购甲.md", "合成合同乙.md", "合成课程丙.md"]
    question = "第一份写了啥？"
    state = _state_with_reordered_presentation(paths)
    _signals, event, decision = _scope(question, state, focus=paths[1])
    resolved_target = decision.result_scope_paths[0]
    now = datetime.now()
    repo_state = SimpleNamespace(
        paths=paths,
        docs=["甲高相关证据", "乙高相关证据", "丙没有索引片段"],
        chunk_paths=paths[:2],
        chunk_texts=["甲高相关证据", "乙高相关证据"],
        chunk_file_times=[now, now],
        chunk_embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=float),
        chunk_meta=[
            {"chunk_id": 0, "start": 0, "end": 7},
            {"chunk_id": 0, "start": 0, "end": 7},
        ],
    )

    materials = build_retrieval_materials(
        question=question,
        search_query=question,
        context_anchor="",
        flags={
            "skip_retrieval": False,
            "is_inventory_query": False,
            "inventory_target_type": None,
        },
        repo_state=repo_state,
        model_emb=_EmbeddingStub(),
        logger=_LoggerStub(),
        current_focus_file=resolved_target,
        event=event,
        allowed_paths=set(decision.result_scope_paths),
    )

    assert resolved_target == paths[2]
    assert materials["current_focus_file"] == resolved_target
    assert materials["relevant_indices"] == []
    assert materials["context_text"] == ""


def test_unbounded_topic_shift_still_releases_previous_focus():
    paths = ["合成资料甲.md", "合成资料乙.md"]
    now = datetime.now()
    repo_state = SimpleNamespace(
        paths=paths,
        docs=["甲主题", "乙主题"],
        chunk_paths=paths,
        chunk_texts=["甲主题", "乙主题"],
        chunk_file_times=[now, now],
        chunk_embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=float),
        chunk_meta=[
            {"chunk_id": 0, "start": 0, "end": 3},
            {"chunk_id": 0, "start": 0, "end": 3},
        ],
    )

    materials = build_retrieval_materials(
        question="那换个主题？",
        search_query="新的合成主题",
        context_anchor="",
        flags={
            "skip_retrieval": False,
            "is_inventory_query": False,
            "inventory_target_type": None,
        },
        repo_state=repo_state,
        model_emb=_EmbeddingStub(),
        logger=_LoggerStub(),
        current_focus_file=paths[0],
        event=SimpleNamespace(name="content_followup"),
        allowed_paths=None,
    )

    assert materials["current_focus_file"] is None
