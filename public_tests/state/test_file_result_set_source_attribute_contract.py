from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from app.dialog.question_scope import analyze_question_signals, decide_file_result_set_scope
from app.dialog.result_set import enforce_file_result_set_attribute_coverage
from app.dialog.state_machine import ConversationState, detect_dialog_event
from app.chat_state_helpers import update_state_after_retrieval_answer
from app.retrieval_flow.materials import build_retrieval_materials, build_safe_final_prompt
from app.retrieval_flow.query import build_search_query
from retrieval.search_intent import determine_query_flags
from retrieval.attribute_evidence import (
    build_requested_attribute_query,
    classify_requested_attribute_kind,
)


ACTIVE_PATHS = ["policy_a.pdf", "vendor_b.pdf", "guide_c.pdf"]
FIVE_DOMAIN_PATHS = [
    "recruitment_criteria.md",
    "contract_draft.md",
    "procurement_guide.md",
    "project_report.md",
    "vendor_manual.md",
]


class _CaptureLogger:
    def __init__(self):
        self.messages: list[str] = []

    def debug(self, message: str):
        self.messages.append(message)

    info = debug
    warning = debug
    error = debug


class _EmbeddingStub:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0] for _text in texts], dtype=float)


def _state_after_collection_summary() -> ConversationState:
    return ConversationState(
        last_user_question="是关于什么的？",
        last_route="normal_retrieval",
        last_content_user_question="是关于什么的？",
        last_content_route="normal_retrieval",
        last_effective_search_query="公共服务 指南 规范",
        last_answer_text="这些材料是对三个不同主题的概括。",
        last_answer_type=None,
        last_result_set_items=list(ACTIVE_PATHS),
        last_result_set_entity_type="文件",
        last_result_set_summary_text="这些材料是对三个不同主题的概括。",
        last_result_set_summary_level=1,
        last_result_set_selectable=False,
    )


@pytest.mark.parametrize(
    "question",
    [
        "是官方的吗？",
        "这些是官方文件吗？",
        "谁发布的？",
        "发布机构呢？",
        "是谁出的？",
        "这些是哪家机构出的？",
        "来源是什么？",
    ],
)
def test_source_attribute_followup_inherits_nonselectable_active_file_scope(question):
    state = _state_after_collection_summary()
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

    assert classify_requested_attribute_kind(question) == "source"
    assert event.name == "result_set_followup"
    assert event.merged_query is None
    assert scope.result_scope_paths == tuple(ACTIVE_PATHS)
    assert scope.query_result_set_items == tuple(ACTIVE_PATHS)
    assert scope.query_result_set_entity == "文件"


@pytest.mark.parametrize(
    ("question", "attribute_kind"),
    [
        ("都是指南吗？", "document_property"),
        ("这些年份一样吗？", None),
        ("发布日期呢？", None),
    ],
)
def test_result_set_scope_resolution_does_not_depend_on_source_classification(
    question,
    attribute_kind,
):
    state = _state_after_collection_summary()
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

    assert classify_requested_attribute_kind(question) == attribute_kind
    assert event.name == "result_set_followup"
    assert scope.result_scope_paths == tuple(ACTIVE_PATHS)
    assert scope.query_result_set_items == tuple(ACTIVE_PATHS)


def test_collection_summary_answer_keeps_file_scope_valid_for_the_next_attribute_turn():
    state = ConversationState(
        last_user_question="有哪些文件？",
        last_route="repo_meta",
        last_content_user_question="有哪些文件？",
        last_content_route="repo_meta",
        last_answer_text="\n".join(
            f"{index}. {path}" for index, path in enumerate(ACTIVE_PATHS, 1)
        ),
        last_answer_type="enumeration_file",
        last_result_set_items=list(ACTIVE_PATHS),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
    )
    summary_question = "是关于什么的？"
    summary_event = detect_dialog_event(summary_question, state, _CaptureLogger())
    summary_signals = analyze_question_signals(
        summary_question,
        last_effective_search_query=state.last_effective_search_query,
    )
    summary_scope = decide_file_result_set_scope(
        summary_question,
        signals=summary_signals,
        state=state,
        current_focus_file=None,
        event_name=summary_event.name,
    )
    update_state_after_retrieval_answer(
        state,
        summary_question,
        "这些材料分别讨论三个不同主题。",
        _CaptureLogger(),
        event_name=summary_event.name,
        question_signals=summary_signals,
        scope_decision=summary_scope,
    )

    assert state.last_answer_type is None
    assert state.last_result_set_items == ACTIVE_PATHS
    assert state.last_result_set_entity_type == "文件"

    next_question = "是官方的吗？"
    next_event = detect_dialog_event(next_question, state, _CaptureLogger())
    next_signals = analyze_question_signals(
        next_question,
        last_effective_search_query=state.last_effective_search_query,
    )
    next_scope = decide_file_result_set_scope(
        next_question,
        signals=next_signals,
        state=state,
        current_focus_file=None,
        event_name=next_event.name,
    )
    assert next_event.name == "result_set_followup"
    assert next_scope.result_scope_paths == tuple(ACTIVE_PATHS)


def test_source_answer_keeps_five_file_scope_for_document_property_followup():
    state = ConversationState(
        last_user_question="是关于什么的？",
        last_route="normal_retrieval",
        last_content_user_question="是关于什么的？",
        last_content_route="normal_retrieval",
        last_effective_search_query="材料主题",
        last_answer_text="这些材料分别讨论不同主题。",
        last_result_set_items=list(FIVE_DOMAIN_PATHS),
        last_result_set_entity_type="文件",
        last_result_set_summary_text="这些材料分别讨论不同主题。",
        last_result_set_summary_level=1,
        last_result_set_selectable=False,
    )
    source_question = "是官方的吗？"
    source_event = detect_dialog_event(source_question, state, _CaptureLogger())
    source_signals = analyze_question_signals(
        source_question,
        last_effective_search_query=state.last_effective_search_query,
    )
    source_scope = decide_file_result_set_scope(
        source_question,
        signals=source_signals,
        state=state,
        current_focus_file=None,
        event_name=source_event.name,
    )
    source_answer = "\n".join(
        f"{index}. {path}：来源主体可由当前证据确认。"
        for index, path in enumerate(FIVE_DOMAIN_PATHS, 1)
    )
    update_state_after_retrieval_answer(
        state,
        source_question,
        source_answer,
        _CaptureLogger(),
        event_name=source_event.name,
        question_signals=source_signals,
        scope_decision=source_scope,
    )

    property_question = "是官方标准吗？"
    property_event = detect_dialog_event(property_question, state, _CaptureLogger())
    property_signals = analyze_question_signals(
        property_question,
        last_effective_search_query=state.last_effective_search_query,
    )
    property_scope = decide_file_result_set_scope(
        property_question,
        signals=property_signals,
        state=state,
        current_focus_file=None,
        event_name=property_event.name,
    )

    assert state.last_result_set_items == FIVE_DOMAIN_PATHS
    assert property_event.name == "result_set_followup"
    assert property_scope.result_scope_paths == tuple(FIVE_DOMAIN_PATHS)
    assert property_scope.query_result_set_items == tuple(FIVE_DOMAIN_PATHS)
    assert classify_requested_attribute_kind(property_question) == "document_property"


@pytest.mark.parametrize(
    "question",
    [
        "“官方”这个词在文件里出现过吗？",
        "给我官方格式的表格",
        "官方要求高血压怎么处理？",
        "重新查所有文件哪些是官方发布的",
        "另外查一下狂犬病暴露处理",
        "当前知识库总共有多少文件？",
        "给我找一下合同里的付款条款",
        "重新搜索所有资料里关于儿童视力的内容",
    ],
)
def test_new_topic_or_repository_wide_request_escapes_old_file_scope(question):
    state = _state_after_collection_summary()
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

    assert event.name != "result_set_followup"
    assert scope.result_scope_paths is None


def test_source_attribute_query_terms_do_not_inherit_previous_summary(monkeypatch):
    def _unexpected_rewrite(*_args, **_kwargs):
        raise AssertionError("scoped source attribute query must not use query rewrite")

    monkeypatch.setattr("app.retrieval_flow.query.rewrite_search_query", _unexpected_rewrite)
    state = _state_after_collection_summary()
    question = "是官方的吗？"
    event = detect_dialog_event(question, state, _CaptureLogger())
    search_query, context_anchor = build_search_query(
        question=question,
        event=event,
        flags=determine_query_flags(question),
        memory_buffer=["AI回答：这些材料主要是公共服务指南或规范。"],
        last_effective_search_query=state.last_effective_search_query,
        last_user_question=state.last_content_user_question,
        last_answer_type=state.last_answer_type,
        last_result_set_items=list(ACTIVE_PATHS),
        last_result_set_entity_type="文件",
        logger=_CaptureLogger(),
        ollama_api_url="http://127.0.0.1:11434/api/generate",
        ollama_model="synthetic-model",
    )

    assert "官方" in search_query
    assert "公共服务" not in search_query
    assert "指南" not in search_query
    assert "规范" not in search_query
    assert all(path not in search_query for path in ACTIVE_PATHS)
    assert context_anchor == ""


@pytest.mark.parametrize(
    ("question", "expected_kind", "expected_query"),
    [
        ("是官方标准吗？", "document_property", "标准"),
        ("这些算正式合同吗？", "document_property", "合同"),
        ("它们属于正式采购规范吗？", "document_property", "规范"),
        ("这些都是岗位说明书吗？", "document_property", "说明书"),
        ("这份标准是谁制定的？", "source", "这份标准是谁制定的"),
        ("这些做法是否符合标准？", None, "做法是否符合标准"),
        ("官方要求中的内容是什么？", None, None),
    ],
)
def test_attribute_classification_separates_provenance_from_document_property(
    question,
    expected_kind,
    expected_query,
):
    assert classify_requested_attribute_kind(question) == expected_kind
    if expected_query is not None:
        assert build_requested_attribute_query(question) == expected_query


def test_document_property_query_keeps_active_scope_without_source_modifier(monkeypatch):
    def _unexpected_rewrite(*_args, **_kwargs):
        raise AssertionError("scoped document-property query must not use query rewrite")

    monkeypatch.setattr("app.retrieval_flow.query.rewrite_search_query", _unexpected_rewrite)
    state = _state_after_collection_summary()
    question = "是官方标准吗？"
    event = detect_dialog_event(question, state, _CaptureLogger())
    search_query, context_anchor = build_search_query(
        question=question,
        event=event,
        flags=determine_query_flags(question),
        memory_buffer=["AI回答：这些材料由不同主体发布。"],
        last_effective_search_query=state.last_effective_search_query,
        last_user_question=state.last_content_user_question,
        last_answer_type=state.last_answer_type,
        last_result_set_items=list(ACTIVE_PATHS),
        last_result_set_entity_type="文件",
        logger=_CaptureLogger(),
        ollama_api_url="http://127.0.0.1:11434/api/generate",
        ollama_model="synthetic-model",
    )

    assert event.name == "result_set_followup"
    assert search_query == "标准"
    assert "官方" not in search_query
    assert all(path not in search_query for path in ACTIVE_PATHS)
    assert context_anchor == ""


def _repo_state_with_source_evidence():
    now = datetime.now()
    outside = "outside_d.pdf"
    chunk_paths = [
        ACTIVE_PATHS[0], ACTIVE_PATHS[0],
        ACTIVE_PATHS[1], ACTIVE_PATHS[1],
        ACTIVE_PATHS[2],
        outside,
    ]
    chunk_texts = [
        "事项背景说明。",
        "发布机构：某政府部门。",
        "产品背景说明。",
        "发布机构：某公司A。",
        "正文未给出发布、制定或署名机构。",
        "发布机构：某政府部门B。",
    ]
    return SimpleNamespace(
        paths=[*ACTIVE_PATHS, outside],
        docs=["", "", "", ""],
        chunk_paths=chunk_paths,
        chunk_texts=chunk_texts,
        chunk_file_times=[now] * len(chunk_paths),
        chunk_embeddings=np.asarray(
            [
                [0.20, 0.0], [0.95, 0.0],
                [0.15, 0.0], [0.90, 0.0],
                [0.10, 0.0],
                [0.99, 0.0],
            ],
            dtype=float,
        ),
        chunk_meta=[
            {"chunk_id": index, "start": index * 20, "end": index * 20 + len(text)}
            for index, text in enumerate(chunk_texts)
        ],
    )


def test_source_attribute_retrieval_is_bounded_and_covers_every_active_file():
    repo_state = _repo_state_with_source_evidence()
    logger = _CaptureLogger()
    materials = build_retrieval_materials(
        question="是官方的吗？",
        search_query="官方",
        context_anchor="",
        flags=determine_query_flags("是官方的吗？"),
        repo_state=repo_state,
        model_emb=_EmbeddingStub(),
        logger=logger,
        current_focus_file=None,
        event=SimpleNamespace(name="result_set_followup"),
        allowed_paths=tuple(ACTIVE_PATHS),
    )

    selected_paths = [repo_state.chunk_paths[index] for index in materials["relevant_indices"]]
    assert set(selected_paths) == set(ACTIVE_PATHS)
    assert "outside_d.pdf" not in selected_paths
    assert materials["relevant_indices"][:3] == [1, 3, 4]
    assert all(f"文件【{path}】" in materials["context_text"] for path in ACTIVE_PATHS)
    assert "outside_d.pdf" not in materials["context_text"]
    assert any("文件集合属性证据" in message for message in logger.messages)


def test_document_property_retrieval_preserves_five_member_bounded_scope():
    now = datetime.now()
    outside = "outside_policy.md"
    chunk_paths = [*FIVE_DOMAIN_PATHS, outside]
    chunk_texts = [
        "由某公司A招聘团队发布。",
        "由某公司B法务团队提供。",
        "由某机构C采购团队编制。",
        "由某组织D项目团队印发。",
        "由某供应方E提供。",
        "本文件明确认定为正式标准。",
    ]
    repo_state = SimpleNamespace(
        paths=chunk_paths,
        docs=[""] * len(chunk_paths),
        chunk_paths=chunk_paths,
        chunk_texts=chunk_texts,
        chunk_file_times=[now] * len(chunk_paths),
        chunk_embeddings=np.asarray(
            [[0.50 + index * 0.01, 0.0] for index in range(len(chunk_paths))],
            dtype=float,
        ),
        chunk_meta=[
            {"chunk_id": index, "start": index * 20, "end": index * 20 + len(text)}
            for index, text in enumerate(chunk_texts)
        ],
    )

    materials = build_retrieval_materials(
        question="是官方标准吗？",
        search_query="标准",
        context_anchor="",
        flags=determine_query_flags("是官方标准吗？"),
        repo_state=repo_state,
        model_emb=_EmbeddingStub(),
        logger=_CaptureLogger(),
        current_focus_file=None,
        event=SimpleNamespace(name="result_set_followup"),
        allowed_paths=tuple(FIVE_DOMAIN_PATHS),
    )

    selected_paths = [repo_state.chunk_paths[index] for index in materials["relevant_indices"]]
    assert selected_paths[:5] == FIVE_DOMAIN_PATHS
    assert set(selected_paths) == set(FIVE_DOMAIN_PATHS)
    assert outside not in selected_paths
    assert all(
        f"文件【{path}】" in materials["context_text"]
        for path in FIVE_DOMAIN_PATHS
    )
    assert outside not in materials["context_text"]


def test_source_attribute_prompt_requires_evidence_and_complete_tri_state_accounting():
    prompt = build_safe_final_prompt(
        memory_buffer=[],
        current_focus_file=None,
        inventory_candidates_text="",
        context_text=(
            "文件【policy_a.pdf】：发布机构为某政府部门。\n"
            "文件【vendor_b.pdf】：发布机构为某公司A。\n"
            "文件【guide_c.pdf】：正文未给出来源。"
        ),
        timeline_evidence_text="",
        question="是官方的吗？",
        event_name="result_set_followup",
        result_set_items=list(ACTIVE_PATHS),
        result_set_entity_type="文件",
    )

    assert all(path in prompt for path in ACTIVE_PATHS)
    assert "严格按上述顺序覆盖每个文件" in prompt
    assert "不能根据文件名" in prompt
    assert "企业、社会组织或其他机构来源" in prompt
    assert "没有足够证据的文件必须逐项写明无法确认" in prompt


@pytest.mark.parametrize(
    ("question", "context_text"),
    [
        (
            "这些都是正式招聘标准吗？",
            "文件【policy_a.pdf】：由某公司A人力部门发布，用于招聘沟通。",
        ),
        (
            "这些算正式合同吗？",
            "文件【vendor_b.pdf】：由某公司B法务部门提供，供协商时参考。",
        ),
        (
            "这些属于正式采购规范吗？",
            "文件【guide_c.pdf】：由某机构C采购部门编制，记录采购建议。",
        ),
    ],
)
def test_document_property_prompt_does_not_promote_source_evidence(
    question,
    context_text,
):
    prompt = build_safe_final_prompt(
        memory_buffer=[],
        current_focus_file=None,
        inventory_candidates_text="",
        context_text=context_text,
        timeline_evidence_text="",
        question=question,
        event_name="result_set_followup",
        result_set_items=list(ACTIVE_PATHS),
        result_set_entity_type="文件",
    )

    assert "【材料性质判断约束】" in prompt
    assert "来源/provenance 证据与材料性质证据是两个独立维度" in prompt
    assert "只能支持相应的来源或主体事实" in prompt
    assert "不能据此自动判定材料具有正式标准" in prompt
    assert "材料性质无法确认" in prompt


def test_document_property_coverage_fallback_preserves_unknown_for_every_member():
    answer, valid = enforce_file_result_set_attribute_coverage(
        "policy_a.pdf：可以认为是正式标准。",
        ACTIVE_PATHS,
        repository_paths=[*ACTIVE_PATHS, "outside_d.pdf"],
        attribute_kind="document_property",
    )

    assert valid is False
    assert "材料性质暂时都按无法确认处理" in answer
    assert all(path in answer for path in ACTIVE_PATHS)
    assert answer.count("无法从当前证据确认") == len(ACTIVE_PATHS)


def test_generated_source_attribute_answer_must_cover_all_members_and_stay_in_scope():
    complete = (
        "1. policy_a.pdf：可由发布机构证据确认是政府部门发布。\n"
        "2. vendor_b.pdf：证据显示由某公司A发布，不作为政府官方材料。\n"
        "3. guide_c.pdf：当前证据无法确认来源。"
    )
    answer, valid = enforce_file_result_set_attribute_coverage(
        complete,
        ACTIVE_PATHS,
        repository_paths=[*ACTIVE_PATHS, "outside_d.pdf"],
    )
    assert valid is True
    assert answer == complete

    for invalid in (
        "policy_a.pdf：可确认。",
        complete + "\n4. outside_d.pdf：可确认。",
    ):
        answer, valid = enforce_file_result_set_attribute_coverage(
            invalid,
            ACTIVE_PATHS,
            repository_paths=[*ACTIVE_PATHS, "outside_d.pdf"],
        )
        assert valid is False
        assert all(path in answer for path in ACTIVE_PATHS)
        assert "outside_d.pdf" not in answer
        assert answer.count("无法从当前证据确认") == len(ACTIVE_PATHS)
