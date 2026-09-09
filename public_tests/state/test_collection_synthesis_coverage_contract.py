from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from app.chat_state_helpers import update_state_after_retrieval_answer
from app.dialog.question_scope import analyze_question_signals, decide_file_result_set_scope
from app.dialog.repo_meta_rules import (
    extract_content_lookup_target,
    is_collection_context_open_enumeration_request,
    is_explicit_corpus_content_enumeration_request,
)
from app.dialog.result_set import (
    build_structured_generated_enumeration_prompt,
    materialize_structured_generated_result_set,
    render_structured_generated_result_set,
)
from app.dialog.state_machine import ConversationState, detect_dialog_event
from app.chat_loop_handlers.guards import is_simple_retrieval_turn
from app.retrieval_flow.materials import build_retrieval_materials, build_safe_final_prompt
from app.retrieval_flow.query import build_search_query
from retrieval.search_engine import perform_retrieval
from retrieval.search_intent import determine_query_flags


class _CaptureLogger:
    def __init__(self):
        self.messages: list[str] = []

    def debug(self, message: str):
        self.messages.append(message)

    def info(self, message: str):
        self.messages.append(message)

    def warning(self, message: str):
        self.messages.append(message)


class _EmbeddingStub:
    def encode(self, _texts):
        return np.asarray([[1.0, 0.0]], dtype=float)


def _file_result_state(paths: list[str]) -> ConversationState:
    answer = "\n".join(f"{index}. {path}" for index, path in enumerate(paths, 1))
    return ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_content_user_question="这些材料是关于什么的？",
        last_effective_search_query="合成集合主题",
        last_answer_text=answer,
        last_answer_preview=answer,
        last_result_set_items=list(paths),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
    )


def _event_and_scope(question: str, paths: list[str]):
    state = _file_result_state(paths)
    event = detect_dialog_event(question, state, _CaptureLogger())
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name=event.name,
    )
    return state, event, scope


@pytest.mark.parametrize(
    ("question", "target", "paths"),
    [
        ("总结一下有哪些岗位？", "岗位", ["招聘甲.md", "招聘乙.md"]),
        ("总结一下有哪些履约条款？", "履约条款", ["合同甲.md", "合同乙.md"]),
        ("总结一下有哪些采购批次？", "采购批次", ["采购甲.md", "采购乙.md"]),
        ("总结一下有哪些病？", "病", ["资料甲.md", "资料乙.md"]),
    ],
)
def test_subjectless_collection_classification_hands_full_scope_and_target_to_retrieval(
    question,
    target,
    paths,
):
    state, event, scope = _event_and_scope(question, paths)

    assert event.name == "synthesis_request"
    assert event.content_target == target
    assert scope.result_scope_paths == tuple(paths)
    assert scope.query_result_set_items == tuple(paths)

    search_query, _context_anchor = build_search_query(
        question=question,
        event=event,
        flags=determine_query_flags(question),
        memory_buffer=[],
        last_effective_search_query=state.last_effective_search_query,
        last_user_question=state.last_content_user_question,
        last_result_set_items=list(scope.query_result_set_items or ()),
        last_result_set_entity_type=scope.query_result_set_entity,
        logger=_CaptureLogger(),
        ollama_api_url="http://127.0.0.1:9",
        ollama_model="synthetic-model",
    )

    assert search_query == target


@pytest.mark.parametrize(
    ("question", "target"),
    [
        ("有哪些技术栈？", "技术栈"),
        ("涉及哪些学历要求？", "学历要求"),
        ("包含哪些期限？", "期限"),
        ("提到了哪些设备？", "设备"),
        ("讲到了哪些技能？", "技能"),
        ("讲了哪些技能？", "技能"),
        ("说了哪些费用？", "费用"),
        ("列出了哪些履约条款？", "履约条款"),
        ("阐明了哪些验收标准？", "验收标准"),
        ("涵盖哪些产品类别？", "产品类别"),
        ("都有什么验收条件？", "验收条件"),
        ("这里面有哪些考核方式？", "考核方式"),
        ("这些资料都涉及什么软件？", "软件"),
        ("一共涉及多少种规格？", "规格"),
    ],
)
def test_collection_open_enumeration_relations_share_structured_semantics(
    question,
    target,
):
    paths = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    _state, event, scope = _event_and_scope(question, paths)

    assert extract_content_lookup_target(question) == target
    assert is_collection_context_open_enumeration_request(
        question,
        has_collection_context=True,
    )
    assert event.name == "synthesis_request"
    assert event.merged_query is None
    assert event.content_target == target
    assert scope.result_scope_paths == tuple(paths)
    assert scope.query_result_set_items == tuple(paths)
    assert scope.query_result_set_entity == "文件"
    assert is_simple_retrieval_turn(question, event.name) is False


@pytest.mark.parametrize(
    ("question", "target"),
    [
        ("整个库提到了哪些术语？", "术语"),
        ("所有文档里有哪些项目？", "项目"),
        ("全部资料包含哪些机构？", "机构"),
        ("全库记录了哪些产品？", "产品"),
    ],
)
def test_explicit_corpus_enumeration_overrides_old_subset_and_focus(
    question,
    target,
    monkeypatch,
):
    old_paths = ["旧范围资料.md"]
    corpus_paths = ["资料甲.md", "资料乙.md", "资料丙.md"]
    state = _file_result_state(old_paths)
    state.last_result_set_focus_file = old_paths[0]
    event = detect_dialog_event(
        question,
        state,
        _CaptureLogger(),
        focused_file=old_paths[0],
    )
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=old_paths[0],
        event_name=event.name,
        corpus_paths=corpus_paths,
    )

    assert is_explicit_corpus_content_enumeration_request(question)
    assert event.name == "synthesis_request"
    assert event.content_target == target
    assert scope.clear_current_focus is True
    assert scope.effective_focus_file is None
    assert scope.result_scope_paths == tuple(corpus_paths)
    assert scope.query_result_set_items == tuple(corpus_paths)
    assert scope.query_result_set_entity == "文件"

    def fail_rewrite(*_args, **_kwargs):
        raise AssertionError("structured corpus target must bypass ordinary rewrite")

    monkeypatch.setattr(
        "app.retrieval_flow.query.rewrite_search_query",
        fail_rewrite,
    )
    search_query, _context_anchor = build_search_query(
        question=question,
        event=event,
        flags=determine_query_flags(question),
        memory_buffer=[],
        last_effective_search_query=state.last_effective_search_query,
        last_user_question=state.last_content_user_question,
        last_result_set_items=list(scope.query_result_set_items or ()),
        last_result_set_entity_type=scope.query_result_set_entity,
        logger=_CaptureLogger(),
        ollama_api_url="http://127.0.0.1:9",
        ollama_model="synthetic-model",
    )

    assert search_query == target


def test_explicit_corpus_enumeration_does_not_capture_adjacent_queries():
    state = ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_effective_search_query="旧主题",
    )

    concrete_lookup = detect_dialog_event("查找项目甲", state, _CaptureLogger())
    capability_query = detect_dialog_event("这个库能干什么？", state, _CaptureLogger())

    assert concrete_lookup.name == "action_request"
    assert concrete_lookup.route_hint == "normal_retrieval"
    assert capability_query.name != "synthesis_request"


@pytest.mark.parametrize(
    "question",
    [
        "重新查所有文件哪些是公开发布的？",
        "整个库哪些属于已验收项目？",
    ],
)
def test_explicit_corpus_selection_predicate_is_not_entity_enumeration(question):
    state = _file_result_state(["旧范围资料.md"])
    event = detect_dialog_event(question, state, _CaptureLogger())
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name=event.name,
    )

    assert not is_explicit_corpus_content_enumeration_request(question)
    assert event.name != "synthesis_request"
    assert scope.result_scope_paths is None


def test_explicit_corpus_enumeration_covers_every_indexed_file_with_empty_terms():
    question = "整个库提到了哪些项？"
    corpus_paths = ["项目资料.md", "机构资料.md", "产品资料.md"]
    state = _file_result_state(["旧范围资料.md"])
    state.last_result_set_focus_file = "旧范围资料.md"
    event = detect_dialog_event(
        question,
        state,
        _CaptureLogger(),
        focused_file="旧范围资料.md",
    )
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file="旧范围资料.md",
        event_name=event.name,
        corpus_paths=corpus_paths,
    )
    now = datetime.now()
    repo_state = SimpleNamespace(
        paths=corpus_paths,
        docs=["项目甲", "机构乙", "产品丙"],
        chunk_paths=corpus_paths,
        chunk_texts=["项目甲", "机构乙", "产品丙"],
        chunk_file_times=[now, now, now],
        chunk_embeddings=np.asarray(
            [[1.0, 0.0], [-1.0, 0.0], [0.05, 0.0]],
            dtype=float,
        ),
        chunk_meta=[
            {"chunk_id": 0, "start": 0, "end": 3}
            for _path in corpus_paths
        ],
    )
    logger = _CaptureLogger()
    materials = build_retrieval_materials(
        question=question,
        search_query=event.content_target or "",
        context_anchor="",
        flags=determine_query_flags(question),
        repo_state=repo_state,
        model_emb=_EmbeddingStub(),
        logger=logger,
        current_focus_file=None if scope.clear_current_focus else "旧范围资料.md",
        event=event,
        allowed_paths=set(scope.result_scope_paths or ()),
        content_target=event.content_target,
    )

    retrieved_paths = {
        repo_state.chunk_paths[index]
        for index in materials["relevant_indices"]
    }
    assert retrieved_paths == set(corpus_paths)
    assert materials["current_focus_file"] is None
    assert any("提取到的核心搜索词: []" in message for message in logger.messages)
    assert not any("禁用兜底" in message for message in logger.messages)
    assert any("为 3/3 个活动文件保留起始主题锚点" in message for message in logger.messages)


def test_unreliable_or_duplicate_generated_candidates_are_not_rendered():
    paths = ["项目资料甲.md", "项目资料乙.md"]
    repo_state = SimpleNamespace(
        paths=paths,
        docs=["名称：项目甲", "名称：项目甲"],
    )
    provenance = materialize_structured_generated_result_set(
        {
            "items": [
                {
                    "display_name": "项目甲",
                    "source_paths": [paths[0]],
                    "evidence_text": "名称：项目甲",
                },
                {
                    "display_name": "项目甲",
                    "source_paths": [paths[1]],
                    "evidence_text": "名称：项目甲",
                },
                {
                    "display_name": "猜测项目",
                    "source_paths": [paths[1]],
                    "evidence_text": "不存在的原文",
                },
            ]
        },
        entity_type="项目",
        candidate_paths=paths,
        repo_state=repo_state,
    )

    rendered = render_structured_generated_result_set(provenance)

    assert provenance.reliable is False
    assert "duplicate_binding" in provenance.binding_failures
    assert "evidence_not_exact" in provenance.binding_failures
    assert rendered == "本轮未生成可可靠引用的枚举结果，请重试。"
    assert "猜测项目" not in rendered


def test_collection_open_enumeration_keeps_file_scope_without_promoting_answer_entities():
    paths = [f"合成资料{index}.md" for index in range(1, 6)]
    state, event, scope = _event_and_scope("讲到了哪些技能？", paths)
    signals = analyze_question_signals(
        "讲到了哪些技能？",
        last_effective_search_query=state.last_effective_search_query,
    )

    update_state_after_retrieval_answer(
        state,
        "讲到了哪些技能？",
        "1. 合成技能甲\n2. 合成技能乙",
        _CaptureLogger(),
        event_name=event.name,
        question_signals=signals,
        scope_decision=scope,
    )

    assert state.last_result_set_items == paths
    assert state.last_result_set_entity_type == "文件"
    assert state.last_result_set_selectable is False
    assert state.last_generated_result_items is None


def test_collection_open_enumeration_explicit_file_scope_wins_over_full_set():
    paths = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    _state, event, scope = _event_and_scope(
        "第二份涉及哪些技术？",
        paths,
    )

    assert event.name == "result_set_followup"
    assert event.content_target == "技术"
    assert scope.selected_file_paths == (paths[1],)
    assert scope.result_scope_paths == (paths[1],)
    assert scope.query_result_set_items == (paths[1],)


def test_collection_open_enumeration_current_file_reference_keeps_file_focus():
    paths = ["合成资料甲.md", "合成资料乙.md"]
    state = _file_result_state(paths)
    question = "这个文件包含哪些期限？"
    event = detect_dialog_event(
        question,
        state,
        _CaptureLogger(),
        focused_file=paths[1],
    )
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=paths[1],
        event_name=event.name,
    )

    assert event.name == "content_followup"
    assert event.content_target == "期限"
    assert scope.result_scope_paths == (paths[1],)


@pytest.mark.parametrize(
    "question",
    [
        "Python 涉及哪些数据类型？",
        "新项目涉及哪些技术栈？",
        "采购系统包含哪些接口？",
        "这份新合同提到了哪些费用？",
        "合同列出了哪些期限？",
        "项目阐明了哪些风险？",
        "方案说明了哪些步骤？",
    ],
)
def test_explicit_new_subject_does_not_inherit_old_collection(question):
    paths = ["旧资料甲.md", "旧资料乙.md"]
    _state, event, scope = _event_and_scope(question, paths)

    assert event.name == "unknown"
    assert event.merged_query is None
    assert event.content_target
    assert scope.result_scope_paths is None
    assert scope.query_result_set_items is None


@pytest.mark.parametrize(
    "question",
    [
        "LangChain 在哪份文件里？",
        "是否涉及开源许可？",
        "这些资料有什么用？",
        "第三个章节出现了什么变化？",
        "讲到 Python 的那份文件是哪一个？",
        "哪些文件提到了 X？",
    ],
)
def test_evidence_and_closed_fact_questions_do_not_become_open_enumeration(question):
    assert not is_collection_context_open_enumeration_request(
        question,
        has_collection_context=True,
    )


def test_presentation_preserves_source_authority_for_next_open_enumeration():
    paths = ["合成资料甲.md", "合成资料乙.md", "合成资料丙.md"]
    state = ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_content_user_question="给我个表格吧",
        last_effective_search_query="合成事项",
        last_answer_text="| 事项 | 来源 |",
        last_result_set_items=["合成事项甲", "合成事项乙"],
        last_result_set_entity_type="事项",
        last_result_set_selectable=True,
        last_generated_result_items=["合成事项甲", "合成事项乙"],
        last_generated_result_entity_type="事项",
        last_generated_result_source_candidates=paths,
        last_generated_result_source_hits=[[paths[0]], [paths[1]]],
    )
    question = "还涉及哪些验收条件？"
    event = detect_dialog_event(question, state, _CaptureLogger())
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    scope = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name=event.name,
    )

    assert event.name == "synthesis_request"
    assert event.content_target == "验收条件"
    assert scope.result_scope_paths == tuple(paths)
    assert scope.query_result_set_items == tuple(paths)
    assert scope.query_result_set_entity == "文件"


def test_empty_core_terms_cannot_drop_members_from_bounded_collection_synthesis():
    paths = ["资料甲.md", "资料乙.md", "资料丙.md"]
    state, event, scope = _event_and_scope("总结一下有哪些病？", paths)
    now = datetime.now()
    repo_state = SimpleNamespace(
        paths=paths,
        docs=["合成主题甲", "合成主题乙", "合成主题丙"],
        chunk_paths=paths,
        chunk_texts=["合成主题甲", "合成主题乙", "合成主题丙"],
        chunk_file_times=[now, now, now],
        chunk_embeddings=np.asarray(
            [[1.0, 0.0], [-1.0, 0.0], [0.05, 0.0]],
            dtype=float,
        ),
        chunk_meta=[
            {"chunk_id": 0, "start": 0, "end": 5},
            {"chunk_id": 0, "start": 0, "end": 5},
            {"chunk_id": 0, "start": 0, "end": 5},
        ],
    )
    logger = _CaptureLogger()
    materials = build_retrieval_materials(
        question="总结一下有哪些病？",
        search_query=event.content_target or "",
        context_anchor="",
        flags=determine_query_flags("总结一下有哪些病？"),
        repo_state=repo_state,
        model_emb=_EmbeddingStub(),
        logger=logger,
        current_focus_file=None,
        event=event,
        allowed_paths=set(scope.result_scope_paths or ()),
        content_target=event.content_target,
    )

    retrieved_paths = [repo_state.chunk_paths[index] for index in materials["relevant_indices"]]
    assert set(retrieved_paths) == set(paths)
    assert any("提取到的核心搜索词: []" in message for message in logger.messages)
    assert any("为 3/3 个活动文件保留起始主题锚点" in message for message in logger.messages)


def test_collection_coverage_keeps_leading_topic_anchors_before_dense_appendices():
    paths = ["招聘总览.md", "合同总览.md", "采购总览.md"]
    now = datetime.now()
    chunk_paths = [path for path in paths for _ in range(2)]
    chunk_texts = [
        "核心主题：候选要求。",
        "附录：合成条目一、合成条目二、合成条目三。",
        "核心主题：履约边界。",
        "附录：合成条目四、合成条目五、合成条目六。",
        "核心主题：采购范围。",
        "附录：合成条目七、合成条目八、合成条目九。",
    ]
    repo_state = SimpleNamespace(
        paths=paths,
        docs=["", "", ""],
        chunk_paths=chunk_paths,
        chunk_texts=chunk_texts,
        chunk_file_times=[now] * len(chunk_paths),
        chunk_embeddings=np.asarray(
            [[0.05, 0.0], [0.9, 0.0]] * len(paths),
            dtype=float,
        ),
        chunk_meta=[
            {"chunk_id": index % 2, "start": index * 10, "end": index * 10 + 9}
            for index in range(len(chunk_paths))
        ],
    )

    retrieval = perform_retrieval(
        "总结一下有哪些项？",
        "项",
        repo_state,
        _EmbeddingStub(),
        _CaptureLogger(),
        None,
        allowed_paths=set(paths),
        task_mode="synthesis_request",
        content_target="项",
        ensure_allowed_path_coverage=True,
    )

    assert retrieval["relevant_indices"][:3] == [0, 2, 4]
    assert set(retrieval["relevant_indices"]) == set(range(6))

    logger = _CaptureLogger()
    materials = build_retrieval_materials(
        question="总结一下有哪些项？",
        search_query="项",
        context_anchor="",
        flags=determine_query_flags("总结一下有哪些项？"),
        repo_state=repo_state,
        model_emb=_EmbeddingStub(),
        logger=logger,
        current_focus_file=None,
        event=SimpleNamespace(name="synthesis_request"),
        allowed_paths=set(paths),
        content_target="项",
    )

    assert all(
        materials["context_text"].count(f"文件【{path}】") == 2
        for path in paths
    )
    assert any("每个活动文件最多保留 2 个片段" in message for message in logger.messages)


def test_collection_enumeration_prompt_blocks_source_and_type_bias():
    prompt = build_structured_generated_enumeration_prompt(
        "合成提示",
        entity_type="合成对象",
    )
    final_prompt = build_safe_final_prompt(
        memory_buffer=[],
        current_focus_file=None,
        inventory_candidates_text="",
        context_text="【参考片段】: 合成内容",
        timeline_evidence_text="",
        question="总结一下有哪些合成对象？",
        event_name="synthesis_request",
        result_set_items=["资料甲.md", "资料乙.md"],
    )

    assert "不得让单一来源的高密度附带列表挤掉其他来源" in prompt
    assert "不要把它的属性、条件、原因或措施当作独立对象" in prompt
    assert "优先提取各来源的核心主题" in prompt
    assert "先逐一核对活动集合中的各个来源" in final_prompt
    assert "把附带提及与核心主题明确区分" in final_prompt


@pytest.mark.parametrize(
    "question",
    [
        "总结一下 Python 有哪些特点？",
        "总结一下人工智能的发展。",
    ],
)
def test_independent_synthesis_does_not_mechanically_inherit_old_file_set(question):
    _state, _event, scope = _event_and_scope(
        question,
        ["旧资料甲.md", "旧资料乙.md"],
    )

    assert scope.result_scope_paths is None
    assert scope.query_result_set_items is None


def test_explicit_single_file_synthesis_still_narrows_scope():
    paths = ["资料甲.md", "资料乙.md", "资料丙.md"]
    _state, event, scope = _event_and_scope(
        "只总结第二个文件有哪些风险？",
        paths,
    )

    assert event.name == "result_set_followup"
    assert scope.result_scope_paths == (paths[1],)
    assert scope.query_result_set_items == (paths[1],)
