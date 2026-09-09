from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from app.chat_state_helpers import update_state_after_retrieval_answer
from app.chat_text.file_lookup import (
    looks_like_bare_content_question,
    looks_like_all_items_file_set_content_question,
    looks_like_file_set_content_question,
    looks_like_implicit_file_set_content_question,
)
from app.chat_text.core import normalize_colloquial_question
from app.dialog.question_scope import (
    analyze_question_signals,
    decide_file_result_set_scope,
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


def _file_result_set(paths: list[str], *, focus: str | None = None) -> ConversationState:
    answer = "\n".join(f"{index}. {path}" for index, path in enumerate(paths, 1))
    return ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_content_user_question="当前有哪些文档？",
        last_effective_search_query="合成资料集合",
        last_answer_text=answer,
        last_answer_preview=answer,
        last_answer_type="enumeration_file",
        last_result_set_items=list(paths),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
        last_result_set_focus_file=focus,
    )


def _event_and_scope(
    question: str,
    state: ConversationState,
    *,
    focus: str | None = None,
):
    question = normalize_colloquial_question(question)
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    event = detect_dialog_event(question, state, _LoggerStub(), focused_file=focus)
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=focus,
        event_name=event.name,
    )
    return signals, event, scope


@pytest.mark.parametrize(
    "question",
    [
        "这些合同文档分别讲了什么？",
        "分别介绍一下",
        "总结一下这些采购文件",
        "都是讲了啥？",
        "都讲了什么？",
        "各自讲什么？",
    ],
)
def test_content_followup_binds_the_complete_active_file_set(question):
    paths = ["合成岗位资料.md", "合成合同说明.md", "合成采购记录.md"]
    _signals, event, scope = _event_and_scope(question, _file_result_set(paths))

    assert event.name == "result_set_followup"
    assert scope.result_scope_paths == tuple(paths)
    assert scope.query_result_set_items == tuple(paths)
    assert scope.query_result_set_entity == "文件"


@pytest.mark.parametrize(
    "question",
    [
        "是关于什么的？",
        "是关于啥的？",
        "都是讲什么的？",
        "都讲啥？",
        "这些是干什么的？",
        "这些主要说什么？",
    ],
)
def test_natural_topic_paraphrases_bind_the_complete_enumerated_file_set(question):
    paths = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    _signals, event, scope = _event_and_scope(question, _file_result_set(paths))

    assert event.name == "result_set_followup"
    assert scope.result_scope_paths == tuple(paths)
    assert scope.query_result_set_items == tuple(paths)
    assert scope.query_result_set_entity == "文件"


@pytest.mark.parametrize(
    ("question", "paths"),
    [
        ("是关于啥的？", ["合成岗位说明甲.md", "合成面试记录乙.md"]),
        ("都是讲什么的？", ["合成合同条款甲.md", "合成履约说明乙.md"]),
        ("这些是干什么的？", ["合成采购需求甲.md", "合成验收记录乙.md"]),
    ],
)
def test_topic_paraphrase_behavior_is_domain_neutral(question, paths):
    _signals, event, scope = _event_and_scope(question, _file_result_set(paths))

    assert event.name == "result_set_followup"
    assert scope.result_scope_paths == tuple(paths)


@pytest.mark.parametrize(
    "question",
    [
        "合成条款是关于什么的？",
        "你是干什么的？",
        "这个术语主要说什么？",
    ],
)
def test_explicit_adjacent_subjects_do_not_bind_to_the_active_file_set(question):
    paths = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    _signals, event, scope = _event_and_scope(question, _file_result_set(paths))

    assert event.name != "result_set_followup"
    assert scope.result_scope_paths != tuple(paths)


def test_explicit_single_file_topic_paraphrase_keeps_single_item_scope():
    paths = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    _signals, event, scope = _event_and_scope(
        "第二个文件是关于啥的？",
        _file_result_set(paths),
    )

    assert event.name == "result_set_followup"
    assert scope.result_scope_paths == (paths[1],)
    assert scope.query_result_set_items == (paths[1],)


@pytest.mark.parametrize(
    "question",
    ["讲了什么？", "讲了啥？", "讲的什么？", "讲的啥？", "内容是什么？"],
)
def test_bare_content_followup_binds_the_complete_active_file_set(question):
    paths = ["合成岗位资料.md", "合成合同说明.md", "合成采购记录.md"]
    normalized = normalize_colloquial_question(question)
    signals, event, scope = _event_and_scope(question, _file_result_set(paths))

    assert "啥" not in normalized
    assert looks_like_bare_content_question(normalized) is True
    assert signals.bare_content_question is True
    assert signals.all_items_file_set_content_question is False
    assert event.name == "result_set_followup"
    assert scope.result_scope_paths == tuple(paths)
    assert scope.query_result_set_items == tuple(paths)
    assert scope.query_result_set_entity == "文件"


@pytest.mark.parametrize(
    ("question", "paths"),
    [
        ("讲的啥？", ["合成岗位说明甲.md", "合成面试记录乙.md"]),
        ("写的什么？", ["合成合同条款甲.md", "合成履约说明乙.md"]),
        ("内容是什么？", ["合成采购需求甲.md", "合成验收记录乙.md"]),
    ],
)
def test_bare_collection_content_followup_generalizes_across_public_domains(
    question,
    paths,
):
    signals, event, scope = _event_and_scope(question, _file_result_set(paths))

    assert signals.bare_content_question is True
    assert event.name == "result_set_followup"
    assert event.route_hint == "normal_retrieval"
    assert scope.result_scope_paths == tuple(paths)
    assert scope.query_result_set_items == tuple(paths)


@pytest.mark.parametrize(
    "question",
    [
        "新的合同主题讲的什么？",
        "采购方案写的什么？",
        "所有文档讲的什么？",
    ],
)
def test_subjectful_content_questions_do_not_inherit_the_active_collection(question):
    paths = ["旧资料甲.md", "旧资料乙.md"]
    signals, event, scope = _event_and_scope(question, _file_result_set(paths))

    assert signals.bare_content_question is False
    assert event.name != "result_set_followup"
    assert scope.result_scope_paths != tuple(paths)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("讲了啥？", "讲了什么？"),
        ("写了啥？", "写了什么？"),
        ("说了啥？", "说了什么？"),
        ("内容是啥？", "内容是什么？"),
    ],
)
def test_content_colloquial_normalization_is_constrained_to_terminal_interrogative(
    question,
    expected,
):
    assert normalize_colloquial_question(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "哪些文档提到了合成验收条件？",
        "患者讲了什么？",
        "第二个问题详细说明",
        "第三个讲什么？",
        "这个讲什么？",
        "关于合成主题讲了什么？",
    ],
)
def test_adjacent_questions_do_not_claim_implicit_file_set_content(question):
    assert looks_like_implicit_file_set_content_question(question) is False
    assert looks_like_file_set_content_question(question) is False


@pytest.mark.parametrize(
    "question",
    ["这是啥文件？", "啥时候修改的？", "他干了啥？", "哪个是啥？"],
)
def test_non_content_colloquial_questions_are_not_normalized_or_claimed(question):
    normalized = normalize_colloquial_question(question)

    assert normalized == question
    assert looks_like_bare_content_question(normalized) is False
    signals = analyze_question_signals(
        normalized,
        last_effective_search_query="合成资料集合",
    )
    assert signals.bare_content_question is False


@pytest.mark.parametrize("question", ["这是啥文件？", "啥时候修改的？", "他干了啥？"])
def test_non_content_colloquial_questions_do_not_enter_result_set_content_event(question):
    signals, event, scope = _event_and_scope(question, _file_result_set(["合成资料.md"]))

    assert signals.bare_content_question is False
    assert event.name != "result_set_followup"
    assert scope.result_scope_paths is None


@pytest.mark.parametrize(
    "question",
    [
        "这些岗位资料都讲了什么？",
        "这些合同文档各自讲什么？",
        "每个采购文件大概讲什么？",
        "都是讲了啥？",
        "都讲了什么？",
        "各自讲什么？",
    ],
)
def test_all_items_content_intent_generalizes_across_domains_and_colloquial_forms(
    question,
):
    signals = analyze_question_signals(
        question,
        last_effective_search_query="合成资料集合",
    )

    assert looks_like_all_items_file_set_content_question(question) is True
    assert signals.all_items_file_set_content_question is True


@pytest.mark.parametrize("question", ["第三个讲什么？", "这个讲什么？", "关于合成主题讲了什么？"])
def test_non_all_items_content_questions_do_not_expand_to_the_complete_file_set(question):
    paths = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    signals, _event, scope = _event_and_scope(question, _file_result_set(paths))

    assert signals.all_items_file_set_content_question is False
    assert scope.result_scope_paths != tuple(paths)


def test_implicit_content_followup_prefers_an_already_selected_file():
    paths = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    state = _file_result_set(paths, focus=paths[2])

    _signals, event, scope = _event_and_scope("讲了什么？", state, focus=paths[2])

    assert event.name == "content_followup"
    assert scope.result_scope_paths == (paths[2],)
    assert scope.query_result_set_items == (paths[2],)


def test_colloquial_bare_content_followup_prefers_an_already_selected_file():
    paths = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    state = _file_result_set(paths, focus=paths[2])

    signals, event, scope = _event_and_scope("讲了啥？", state, focus=paths[2])

    assert signals.bare_content_question is True
    assert event.name == "content_followup"
    assert scope.result_scope_paths == (paths[2],)
    assert scope.query_result_set_items == (paths[2],)


@pytest.mark.parametrize("question", ["讲了什么？", "讲了啥？"])
def test_bare_content_without_file_state_does_not_create_result_set_scope(question):
    state = ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
    )

    signals, event, scope = _event_and_scope(question, state)

    assert signals.bare_content_question is True
    assert event.name != "result_set_followup"
    assert scope.result_scope_paths is None
    assert scope.query_result_set_items is None


def test_collective_content_request_overrides_selected_focus_with_complete_file_set():
    paths = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    state = _file_result_set(paths, focus=paths[2])

    signals, event, scope = _event_and_scope("都是讲了啥？", state, focus=paths[2])

    assert signals.all_items_file_set_content_question is True
    assert event.name == "result_set_followup"
    assert scope.result_scope_paths == tuple(paths)
    assert scope.query_result_set_items == tuple(paths)


@pytest.mark.parametrize(
    ("question", "expected_index"),
    [
        ("第三个呢？", 2),
        ("第二个详细说说", 1),
        ("第一个讲什么？", 0),
    ],
)
def test_ordinal_content_requests_resolve_exact_file_identity(question, expected_index):
    paths = ["01_合成资料.pdf", "02_合成记录.pdf", "03_合成说明.pdf"]
    signals, event, scope = _event_and_scope(question, _file_result_set(paths))

    assert signals.explicit_single_file_result_reference is True
    assert event.name == "result_set_followup"
    assert scope.selected_file_paths == (paths[expected_index],)
    assert scope.result_scope_paths == (paths[expected_index],)
    assert scope.selected_result_set_item_turn is True


def test_selected_ordinal_is_written_as_focus_and_inherited_by_detail_followup():
    paths = ["01_合成资料.pdf", "02_合成记录.pdf", "03_合成说明.pdf"]
    state = _file_result_set(paths)
    question = "第三个呢？"
    signals, event, scope = _event_and_scope(question, state)

    update_state_after_retrieval_answer(
        state,
        question,
        "该文件介绍了合成说明。",
        _LoggerStub(),
        event_name=event.name,
        focused_file=paths[2],
        question_signals=signals,
        scope_decision=scope,
    )

    assert state.last_result_set_items == paths
    assert state.last_result_set_focus_file == paths[2]

    _next_signals, next_event, next_scope = _event_and_scope(
        "详细讲下",
        state,
        focus=paths[2],
    )
    assert next_event.name == "content_followup"
    assert next_scope.result_scope_paths == (paths[2],)
    assert next_scope.query_result_set_items == (paths[2],)


def test_file_by_file_content_answer_keeps_original_order_selectable_for_ordinal():
    paths = ["01_合成资料.pdf", "02_合成记录.pdf", "03_合成说明.pdf"]
    state = _file_result_set(paths)
    question = "这些文档分别讲了什么？"
    signals, event, scope = _event_and_scope(question, state)

    update_state_after_retrieval_answer(
        state,
        question,
        "1. 合成主题甲\n2. 合成主题乙\n3. 合成主题丙",
        _LoggerStub(),
        event_name=event.name,
        question_signals=signals,
        scope_decision=scope,
    )

    assert state.last_result_set_items == paths
    assert state.last_result_set_selectable is True
    _signals, _event, ordinal_scope = _event_and_scope("第三个呢？", state)
    assert ordinal_scope.selected_file_paths == (paths[2],)


def test_normal_retrieval_answer_does_not_erase_active_file_set_without_reset_signal():
    paths = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    state = _file_result_set(paths)
    question = "补充说明合成主题"
    signals, _event, scope = _event_and_scope(question, state)

    update_state_after_retrieval_answer(
        state,
        question,
        "这是补充说明。\n来源：合成资料乙.md",
        _LoggerStub(),
        event_name="unknown",
        focused_file=None,
        question_signals=signals,
        scope_decision=scope,
    )

    assert state.last_result_set_items == paths
    assert state.last_result_set_entity_type == "文件"
    assert state.last_result_set_selectable is True


@pytest.mark.parametrize(
    "question",
    [
        "这些文档分别讲了什么？",
        "分别介绍一下",
        "都是讲了啥？",
        "都讲了什么？",
        "各自讲什么？",
    ],
)
def test_file_set_content_retrieval_keeps_evidence_from_every_scoped_file(question):
    paths = ["合成岗位资料.md", "合成合同说明.md", "合成采购记录.md"]
    now = datetime.now()
    repo_state = SimpleNamespace(
        paths=paths,
        docs=["合成甲", "合成乙", "合成丙"],
        chunk_paths=paths,
        chunk_texts=["合成甲", "合成乙", "合成丙"],
        chunk_file_times=[now, now, now],
        chunk_embeddings=np.asarray(
            [[1.0, 0.0], [-1.0, 0.0], [0.1, 0.0]],
            dtype=float,
        ),
        chunk_meta=[
            {"chunk_id": 0, "start": 0, "end": 3},
            {"chunk_id": 0, "start": 0, "end": 3},
            {"chunk_id": 0, "start": 0, "end": 3},
        ],
    )
    state = _file_result_set(paths)
    _signals, event, scope = _event_and_scope(question, state)

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
        current_focus_file=None,
        event=event,
        allowed_paths=set(scope.result_scope_paths or ()),
    )

    retrieved_paths = [
        repo_state.chunk_paths[index] for index in materials["relevant_indices"]
    ]
    assert set(retrieved_paths) == set(paths)
    assert retrieved_paths == paths
