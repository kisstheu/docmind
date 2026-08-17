from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from ai.query_router import route_question
from ai.repo_meta.answering import answer_repo_meta_question
from app.chat_state_helpers import update_state_after_retrieval_answer
from app.chat_text.lookup_answer_main import maybe_build_direct_lookup_answer
from app.chat_text.file_lookup import maybe_build_file_location_answer
from app.dialog.question_scope import analyze_question_signals, decide_file_result_set_scope
from app.dialog.state_machine import ConversationState, detect_dialog_event
from app.domain_dispatch_port import adapt_domain_content_query
from app.domain_host import EmptyDomainHost
from app.retrieval_flow.query import build_search_query
from bootstrap.domain_composition import create_domain_host
from docmind_recruitment_plugin import PLUGIN_ID, RecruitmentJDPlugin
from retrieval.search_engine import perform_retrieval
from retrieval.search_intent import determine_query_flags, is_weak_query


class _CaptureLogger:
    def __init__(self):
        self.messages: list[str] = []

    def debug(self, message: str):
        self.messages.append(message)

    def info(self, message: str):
        self.messages.append(message)

    def warning(self, message: str):
        self.messages.append(message)


class _InventoryResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"response": '{"inventory_listing": true}'}


class _EmbeddingStub:
    def encode(self, _texts):
        return np.asarray([[0.25, 0.0]], dtype=float)


def _file_result_state(paths: list[str]) -> ConversationState:
    return ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_content_user_question="是关于什么的？",
        last_effective_search_query="合成主题概括",
        last_answer_text="这些材料整体围绕一个合成主题。",
        last_answer_preview="这些材料整体围绕一个合成主题。",
        last_answer_type=None,
        last_result_set_items=list(paths),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
        last_result_set_summary_text="这些材料整体围绕一个合成主题。",
        last_result_set_summary_level=1,
    )


def _scope_facts(question: str, state: ConversationState, event_name: str):
    signals = analyze_question_signals(
        question,
        last_effective_search_query=state.last_effective_search_query,
    )
    decision = decide_file_result_set_scope(
        question,
        signals=signals,
        state=state,
        current_focus_file=None,
        event_name=event_name,
    )
    return signals, decision


def test_local_inventory_semantic_maps_to_bounded_repo_meta_action(monkeypatch):
    monkeypatch.setattr(
        "ai.query_router.requests.post",
        lambda *_args, **_kwargs: _InventoryResponse(),
    )

    route_info = route_question(
        "有哪些笔记、文档或资料？",
        "http://127.0.0.1:11434/api/generate",
        "synthetic-model",
        _CaptureLogger(),
    )

    assert route_info == {
        "route": "repo_meta",
        "action": "list_files",
        "semantic_source": "local_model_inventory",
    }


def test_inventory_action_reuses_list_files_without_string_reclassification(monkeypatch):
    paths = ["合成资料甲.md", "合成资料乙.txt"]
    repo_state = SimpleNamespace(
        paths=paths,
        all_files=paths,
        file_times=[datetime.now(), datetime.now()],
    )

    def _must_not_reclassify(*_args, **_kwargs):
        raise AssertionError("structured inventory action must bypass raw-string subtype guessing")

    monkeypatch.setattr(
        "ai.repo_meta.answering.classify_repo_meta_question",
        _must_not_reclassify,
    )

    answer, topic = answer_repo_meta_question(
        "有哪些笔记、文档或资料？",
        repo_state,
        semantic_action="list_files",
    )

    assert topic == "list_files"
    assert all(path in answer for path in paths)
    assert "换个更直接的问法" not in answer


def test_existing_direct_file_list_shortcut_stays_deterministic():
    event = detect_dialog_event("有哪些文件？", ConversationState(), _CaptureLogger())

    assert event.name == "repo_meta_request"
    assert event.route_hint == "repo_meta"
    assert event.content_target is None


@pytest.mark.parametrize(
    ("question", "expected_target"),
    [
        ("这些文件里有哪些岗位？", "岗位"),
        ("这些文件里有哪些履约条款？", "履约条款"),
        ("这些文件里有哪些采购批次？", "采购批次"),
    ],
)
def test_scoped_content_target_becomes_retrieval_query_across_domains(
    monkeypatch,
    question,
    expected_target,
):
    paths = ["合成资料甲.md", "合成资料乙.md"]
    state = _file_result_state(paths)
    event = detect_dialog_event(question, state, _CaptureLogger())
    _signals, decision = _scope_facts(question, state, event.name)

    def _must_not_rewrite(*_args, **_kwargs):
        raise AssertionError("scoped structured content target must not be re-guessed")

    monkeypatch.setattr(
        "app.retrieval_flow.query.rewrite_search_query",
        _must_not_rewrite,
    )
    flags = determine_query_flags(question)
    search_query, _context_anchor = build_search_query(
        question=question,
        event=event,
        flags=flags,
        memory_buffer=[],
        last_effective_search_query=state.last_effective_search_query,
        last_user_question=state.last_content_user_question,
        last_answer_type=state.last_answer_type,
        last_result_set_items=list(decision.query_result_set_items or ()),
        last_result_set_entity_type=decision.query_result_set_entity,
        logger=_CaptureLogger(),
        ollama_api_url="http://127.0.0.1:11434/api/generate",
        ollama_model="synthetic-model",
    )

    assert event.content_target == expected_target
    assert decision.result_scope_paths == tuple(paths)
    assert flags["is_inventory_query"] is True
    assert search_query == expected_target


def test_content_target_without_result_scope_keeps_existing_rewrite_path(monkeypatch):
    calls: list[str] = []

    def _rewrite(question, *_args, **_kwargs):
        calls.append(question)
        return question

    monkeypatch.setattr("app.retrieval_flow.query.rewrite_search_query", _rewrite)
    event = SimpleNamespace(
        name="content_followup",
        route_hint="normal_retrieval",
        merged_query="合成主题 需要哪些条款",
        content_target="条款",
    )

    build_search_query(
        question="需要哪些条款？",
        event=event,
        flags={"skip_retrieval": False, "is_inventory_query": False},
        memory_buffer=[],
        last_effective_search_query="合成主题",
        last_result_set_items=None,
        last_result_set_entity_type=None,
        logger=_CaptureLogger(),
        ollama_api_url="http://127.0.0.1:11434/api/generate",
        ollama_model="synthetic-model",
    )

    assert calls


@pytest.mark.parametrize("content_target", ["岗位", "履约条款", "采购批次"])
def test_structured_content_target_does_not_fall_into_file_locator_shortcut(
    content_target,
):
    repo_state = SimpleNamespace(
        chunk_paths=["合成资料甲.md"],
        chunk_texts=[f"{content_target}包含合成可核对内容。"],
    )

    answer = maybe_build_file_location_answer(
        question=f"这些文件里有哪些{content_target}？",
        search_query=content_target,
        relevant_indices=[0],
        repo_state=repo_state,
        content_target=content_target,
        allow_followup_inference=True,
    )

    assert answer is None


def test_short_query_is_only_strengthened_by_bounded_content_context():
    assert is_weak_query("JD？", ["JD"]) is True
    assert is_weak_query(
        "这些文件里有哪些JD？",
        ["JD"],
        has_bounded_scope=True,
        has_content_enumeration_intent=True,
    ) is False


def test_scoped_short_content_target_can_retrieve_but_fresh_query_stays_guarded():
    now = datetime.now()
    repo_state = SimpleNamespace(
        paths=["合成资料甲.md"],
        chunk_paths=["合成资料甲.md"],
        chunk_texts=["该合成记录明确包含 JD 字段。"],
        chunk_file_times=[now],
        chunk_embeddings=np.asarray([[1.0, 0.0]], dtype=float),
        docs=["该合成记录明确包含 JD 字段。"],
    )

    scoped = perform_retrieval(
        "这些文件里有哪些JD？",
        "JD",
        repo_state,
        _EmbeddingStub(),
        _CaptureLogger(),
        None,
        allowed_paths={"合成资料甲.md"},
        content_target="JD",
        task_mode="content_followup",
    )
    fresh = perform_retrieval(
        "JD？",
        "JD",
        repo_state,
        _EmbeddingStub(),
        _CaptureLogger(),
        None,
    )

    assert scoped["relevant_indices"] == [0]
    assert fresh["relevant_indices"] == []


@pytest.mark.parametrize(
    ("question", "expected_target"),
    [
        ("有哪些岗位", "岗位"),
        ("有哪些职位", "职位"),
        ("有哪些 JD", "jd"),
        ("有哪些职位描述", "职位描述"),
        ("有哪些jd", "jd"),
        ("有什么 JD", "jd"),
    ],
)
def test_fresh_content_listing_keeps_structured_target_for_optional_adapter(
    question: str,
    expected_target: str,
):
    event = detect_dialog_event(question, ConversationState(), _CaptureLogger())

    assert event.name == "unknown"
    assert event.content_target == expected_target


def test_recruitment_jd_adapter_reuses_core_listing_without_global_alias():
    now = datetime.now()
    source_term_groups = (("岗位职责", "任职要求"),)
    source_texts = (
        "岗位职责：维护合成服务。\n任职要求：熟悉 Python。",
        "岗位职责：维护合成测试。\n任职要求：熟悉 SQL。",
    )
    repo_state = SimpleNamespace(
        paths=["合成资料甲.md", "合成资料乙.md"],
        docs=list(source_texts),
        chunk_paths=["合成资料甲.md", "合成资料乙.md"],
        chunk_texts=[
            "岗位：合成服务开发工程师\n职位：合成服务开发工程师\n职位描述：维护合成服务。",
            "岗位：合成测试工程师\n职位：合成测试工程师\n职位描述：维护合成测试。",
        ],
        chunk_file_times=[now, now],
        chunk_embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=float),
    )
    host = create_domain_host(
        plugin=RecruitmentJDPlugin(),
        expected_plugin_id=PLUGIN_ID,
    )
    effective_queries: dict[str, str] = {}
    retrievals: dict[str, list[int]] = {}

    for question in ("有哪些岗位", "有哪些职位", "有哪些 JD", "有哪些职位描述"):
        event = detect_dialog_event(question, ConversationState(), _CaptureLogger())
        content_target = event.content_target or ""
        adapted = adapt_domain_content_query(
            host,
            question=question,
            content_target=content_target,
            source_term_groups=source_term_groups,
        )
        effective_query = adapted or content_target
        effective_queries[question] = effective_query
        retrieval = perform_retrieval(
            question,
            effective_query,
            repo_state,
            _EmbeddingStub(),
            _CaptureLogger(),
            None,
            allowed_paths=set(repo_state.paths),
            content_target=effective_query,
        )
        retrievals[question] = retrieval["relevant_indices"]

    assert effective_queries == {
        "有哪些岗位": "岗位",
        "有哪些职位": "职位",
        "有哪些 JD": "岗位",
        "有哪些职位描述": "职位描述",
    }
    assert all(retrievals[question] == [0, 1] for question in retrievals)

    jd_answer = maybe_build_direct_lookup_answer(
        question="有哪些 JD",
        search_query=effective_queries["有哪些 JD"],
        relevant_indices=retrievals["有哪些 JD"],
        repo_state=repo_state,
        allow_followup_inference=True,
        force_local_evidence=True,
    )
    position_answer = maybe_build_direct_lookup_answer(
        question="有哪些岗位",
        search_query=effective_queries["有哪些岗位"],
        relevant_indices=retrievals["有哪些岗位"],
        repo_state=repo_state,
        allow_followup_inference=True,
        force_local_evidence=True,
    )
    assert jd_answer == position_answer
    assert "合成服务开发工程师" in (jd_answer or "")
    assert "合成测试工程师" in (jd_answer or "")


def test_recruitment_jd_canonical_target_reaches_direct_lookup_fallback():
    now = datetime.now()
    paths = ["合成资料甲.md", "合成资料乙.md"]
    repo_state = SimpleNamespace(
        paths=paths,
        chunk_paths=paths,
        chunk_texts=[
            "岗位职责\n岗位一：合成服务开发工程师\n任职要求：熟悉 Python。",
            "岗位职责\n岗位二：合成测试工程师\n任职要求：熟悉 SQL。",
        ],
        chunk_file_times=[now, now],
        chunk_embeddings=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=float),
    )
    host = create_domain_host(
        plugin=RecruitmentJDPlugin(),
        expected_plugin_id=PLUGIN_ID,
    )
    adapted = adapt_domain_content_query(
        host,
        question="有哪些 JD",
        content_target="JD",
        source_term_groups=(("岗位职责", "任职要求"),),
    )

    retrieval = perform_retrieval(
        "有哪些 JD",
        adapted,
        repo_state,
        _EmbeddingStub(),
        _CaptureLogger(),
        None,
        allowed_paths=set(paths),
        content_target=adapted,
    )

    assert retrieval["relevant_indices"] == [0, 1]

    answer = maybe_build_direct_lookup_answer(
        question="有哪些 JD",
        search_query=adapted,
        relevant_indices=retrieval["relevant_indices"],
        repo_state=repo_state,
        allow_followup_inference=True,
        force_local_evidence=True,
        canonical_content_target=adapted,
    )

    assert answer is not None
    assert "岗位职责" not in (answer or "")
    assert "岗位一：合成服务开发工程师" in (answer or "")
    assert "岗位二：合成测试工程师" in (answer or "")


def test_empty_host_and_unrelated_sources_do_not_enable_jd_alias():
    contract_term_groups = (
        ("合同条款", "履约要求"),
        ("采购需求", "供应商资格"),
    )

    assert adapt_domain_content_query(
        EmptyDomainHost(),
        question="有哪些 JD",
        content_target="JD",
        source_term_groups=(("岗位职责", "任职要求"),),
    ) is None
    assert adapt_domain_content_query(
        create_domain_host(
            plugin=RecruitmentJDPlugin(),
            expected_plugin_id=PLUGIN_ID,
        ),
        question="有哪些 JD",
        content_target="JD",
        source_term_groups=contract_term_groups,
    ) is None


def test_zero_hit_scoped_followup_preserves_parent_file_result_set():
    paths = ["合成合同.md", "合成采购记录.md"]
    state = _file_result_state(paths)
    question = "这些文件里有哪些未出现字段？"
    event = detect_dialog_event(question, state, _CaptureLogger())
    signals, decision = _scope_facts(question, state, event.name)

    update_state_after_retrieval_answer(
        state,
        question,
        "本轮没有检索到可用证据，建议换更具体关键词或指定文件名重试。",
        _CaptureLogger(),
        event_name=event.name,
        focused_file=None,
        question_signals=signals,
        scope_decision=decision,
    )

    assert state.last_answer_type is None
    assert state.last_result_set_items == paths
    assert state.last_result_set_entity_type == "文件"
    assert state.last_result_set_selectable is True

    next_question = "这些文件里有哪些履约条款？"
    next_event = detect_dialog_event(next_question, state, _CaptureLogger())
    _next_signals, next_decision = _scope_facts(
        next_question,
        state,
        next_event.name,
    )
    assert next_decision.result_scope_paths == tuple(paths)
