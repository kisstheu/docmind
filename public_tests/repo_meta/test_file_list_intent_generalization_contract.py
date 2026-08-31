from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from ai.repo_meta.answering import answer_repo_meta_question
from ai.repo_meta.classifier import (
    classify_repo_meta_question,
    extract_topic_from_list_request,
    parse_file_list_request,
)
from app.chat_state_helpers import update_state_after_local_answer
from app.dialog.state_machine import ConversationState, detect_dialog_event
from retrieval.repo_index_types import RepoState


GLOBAL_LIST_BASE_FORMS = (
    "有哪些文档？",
    "知识库里有什么文件？",
    "列出全部资料。",
)

DISCOURSE_LEAD_INS = (
    "",
    "那，",
    "那现在，",
    "好，那目前 ",
    "所以现在，",
    "嗯，那当前，",
    "顺便问一下，",
    "接着说，",
)

OPAQUE_TOPICS = (
    "蓝鲸计划",
    "X17协议",
    "Orchid-42",
    "甲类",
)

EXPLICIT_TOPIC_TEMPLATES = (
    "有哪些关于{topic}的文档？",
    "有哪些{topic}相关资料？",
    "关于{topic}的文件有哪些？",
)

AMBIGUOUS_POSTFIX_TOPIC_CASES = (
    ("顺便问一下", "蓝鲸计划", "资料"),
    ("接着说", "Orchid-42", "资料"),
    ("我再看看", "甲类", "文件"),
)


class _LoggerStub:
    def debug(self, *_args, **_kwargs):
        return None

    info = debug
    warning = debug
    error = debug


class _SyntheticEmbedding:
    """Deterministic local vectors; no model or network call is involved."""

    def encode(self, texts):
        return np.array(
            [[1.0, 0.0] if "蓝鲸计划" in text else [0.0, 1.0] for text in texts]
        )


def _repo_state(
    paths: list[str],
    *,
    shadow_tags: list[str] | None = None,
) -> RepoState:
    now = datetime(2026, 8, 31, 12, 0, 0)
    tags = shadow_tags or ["" for _ in paths]
    docs = [f"{path} 的合成内容" for path in paths]
    return RepoState(
        docs=docs,
        doc_records=[
            {"path": path, "doc": doc, "shadow_tags": tag, "scene_tags": []}
            for path, doc, tag in zip(paths, docs, tags)
        ],
        paths=list(paths),
        file_times=[now for _ in paths],
        file_info_list=list(paths),
        chunk_texts=docs,
        chunk_paths=list(paths),
        chunk_meta=[
            {"chunk_id": index, "start": 0, "end": len(doc)}
            for index, doc in enumerate(docs)
        ],
        chunk_file_times=[now for _ in paths],
        all_files=[Path("synthetic_notes") / path for path in paths],
        embeddings=np.zeros((len(paths), 2)),
        chunk_embeddings=np.zeros((len(paths), 2)),
        earliest_note=paths[-1] if paths else "",
        latest_note=paths[0] if paths else "",
    )


@pytest.mark.parametrize(
    ("lead_in", "base_request"),
    [
        (lead_in, base_request)
        for lead_in in DISCOURSE_LEAD_INS
        for base_request in GLOBAL_LIST_BASE_FORMS
    ],
)
def test_global_file_list_is_invariant_under_discourse_lead_in(lead_in, base_request):
    question = f"{lead_in}{base_request}"

    assert parse_file_list_request(question) == ""
    assert extract_topic_from_list_request(question) == ""
    assert classify_repo_meta_question(question) == "list_files"
    assert classify_repo_meta_question(question) != "list_files_by_topic"


@pytest.mark.parametrize(
    ("lead_in", "topic", "template"),
    [
        (lead_in, topic, template)
        for lead_in in DISCOURSE_LEAD_INS
        for topic in OPAQUE_TOPICS
        for template in EXPLICIT_TOPIC_TEMPLATES
    ],
)
def test_explicit_topic_is_invariant_under_discourse_lead_in(lead_in, topic, template):
    base_request = template.format(topic=topic)
    question = f"{lead_in}{base_request}"

    assert parse_file_list_request(base_request) == topic
    assert parse_file_list_request(question) == topic
    assert extract_topic_from_list_request(question) == topic
    assert classify_repo_meta_question(question) == "list_files_by_topic"


@pytest.mark.parametrize(
    ("lead_in", "topic", "object_term"),
    AMBIGUOUS_POSTFIX_TOPIC_CASES,
)
def test_unsegmented_postfix_topic_does_not_swallow_lead_in(
    lead_in,
    topic,
    object_term,
):
    question = f"{lead_in}{topic}相关有哪些{object_term}？"
    forbidden_topic = f"{lead_in}{topic}"

    assert parse_file_list_request(question) == ""
    assert extract_topic_from_list_request(question) == ""
    assert parse_file_list_request(question) != forbidden_topic
    assert classify_repo_meta_question(question) == "list_files"


@pytest.mark.parametrize(
    ("question", "forbidden_topic"),
    [
        ("顺便问一下，有哪些文档？", "顺便问一下"),
        ("接着说，现在库里有什么资料？", "接着说"),
        ("我再看看，知识库里有哪些文件？", "我再看看"),
    ],
)
def test_unmarked_lead_in_never_becomes_topic(question, forbidden_topic):
    parsed = parse_file_list_request(question)

    assert parsed == ""
    assert parsed != forbidden_topic
    assert classify_repo_meta_question(question) == "list_files"


@pytest.mark.parametrize(
    ("question", "classification"),
    [
        ("最近修改了哪些文件？", "time"),
        ("有多少个文件？", "count"),
        ("有哪些文件类型？", "list_files"),
        ("知识库有多大？", None),
    ],
)
def test_adjacent_repo_meta_intents_remain_unchanged(question, classification):
    assert classify_repo_meta_question(question) == classification


def test_core_acceptance_chain_from_capability_to_global_file_list():
    question = "那现在有哪些文档？"
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    state = ConversationState(
        last_user_question="能干啥？",
        last_route="system_capability",
        last_answer_text="可以检索和整理本地资料。",
    )

    event = detect_dialog_event(question, state, _LoggerStub())
    classification = classify_repo_meta_question(
        question,
        last_user_question=state.last_user_question,
        last_local_topic=state.last_local_topic,
    )
    answer, local_topic = answer_repo_meta_question(
        question,
        _repo_state(paths),
        last_user_question=state.last_user_question,
        last_local_topic=state.last_local_topic,
    )
    update_state_after_local_answer(
        state,
        question=question,
        answer=answer,
        route="repo_meta",
        local_topic=local_topic,
        is_content_answer=True,
    )

    assert event.name == "repo_meta_request"
    assert event.route_hint == "repo_meta"
    assert classification == "list_files"
    assert local_topic == "list_files"
    assert [line for line in answer.splitlines() if line[:1].isdigit()] == [
        "1. 资料甲.md",
        "2. 记录乙.txt",
        "3. 说明丙.pdf",
    ]
    assert "没有明显命中" not in answer
    assert "那现在" not in answer
    assert state.last_result_set_items == paths
    assert state.last_result_set_selectable is True


def test_core_acceptance_chain_preserves_explicit_topic_after_capability():
    question = "那现在有哪些关于蓝鲸计划的文档？"
    paths = ["蓝鲸计划说明.md", "蓝鲸计划记录.txt", "其他资料.txt"]
    state = ConversationState(
        last_user_question="能干啥？",
        last_route="system_capability",
        last_answer_text="可以检索和整理本地资料。",
    )

    event = detect_dialog_event(question, state, _LoggerStub())
    parsed_topic = parse_file_list_request(question)
    classification = classify_repo_meta_question(
        question,
        last_user_question=state.last_user_question,
        last_local_topic=state.last_local_topic,
    )
    answer, local_topic = answer_repo_meta_question(
        question,
        _repo_state(
            paths,
            shadow_tags=["蓝鲸计划", "蓝鲸计划", "其他主题"],
        ),
        model_emb=_SyntheticEmbedding(),
        last_user_question=state.last_user_question,
        last_local_topic=state.last_local_topic,
    )
    update_state_after_local_answer(
        state,
        question=question,
        answer=answer,
        route="repo_meta",
        local_topic=local_topic,
        is_content_answer=True,
    )

    assert event.name == "repo_meta_request"
    assert event.route_hint == "repo_meta"
    assert parsed_topic == "蓝鲸计划"
    assert classification == "list_files_by_topic"
    assert local_topic == "list_files_by_topic"
    assert "蓝鲸计划说明.md" in answer
    assert "蓝鲸计划记录.txt" in answer
    assert "其他资料.txt" not in answer
    assert state.last_result_set_items is not None
    assert set(state.last_result_set_items) == {"蓝鲸计划说明.md", "蓝鲸计划记录.txt"}
    assert state.last_result_set_selectable is True
