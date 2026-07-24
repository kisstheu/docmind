from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ai.repo_meta.answering import answer_repo_meta_question
from ai.repo_meta.classifier import (
    classify_repo_meta_question,
    extract_topic_from_list_request,
    parse_file_list_request,
)
from app import chat_loop as chat_runtime
import app.chat_loop_parts.runner as chat_runner
from app.chat_state_helpers import update_state_after_local_answer
from app.dialog.result_set import resolve_file_result_set_selection
from app.dialog.state_machine import ConversationState, detect_dialog_event
from app.domain_host.host import EmptyDomainHost
from retrieval.repo_index_types import RepoState


class _LoggerStub:
    def debug(self, *_args, **_kwargs):
        return None

    info = debug
    warning = debug
    error = debug


class _EmbeddingStub:
    def encode(self, texts):
        return np.array([[0.8, 0.0] for _ in texts])


class _ModelsStub:
    def __init__(self):
        self.calls: list[str] = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(contents)
        return SimpleNamespace(text="资料甲记录检索，记录乙说明路由，说明丙描述归档。")


class _ClientStub:
    def __init__(self):
        self.models = _ModelsStub()


def _repo_state(paths: list[str]):
    now = datetime(2026, 7, 24, 12, 0, 0)
    return SimpleNamespace(
        paths=list(paths),
        all_files=[Path("synthetic_notes") / path for path in paths],
        file_times=[now for _ in paths],
        docs=[f"{path} 的合成内容" for path in paths],
        doc_records=[
            {
                "path": path,
                "doc": f"{path} 的合成内容",
                "shadow_tags": "",
                "scene_tags": [],
            }
            for path in paths
        ],
    )


def _indexed_repo_state(paths: list[str]) -> RepoState:
    now = datetime(2026, 7, 24, 12, 0, 0)
    chunks = [f"{path} 的合成知识内容。" for path in paths]
    return RepoState(
        docs=chunks,
        doc_records=[
            {"path": path, "doc": chunk, "shadow_tags": "", "scene_tags": []}
            for path, chunk in zip(paths, chunks)
        ],
        paths=list(paths),
        file_times=[now for _ in paths],
        file_info_list=list(paths),
        chunk_texts=chunks,
        chunk_paths=list(paths),
        chunk_meta=[
            {"chunk_id": 0, "start": 0, "end": len(chunk)}
            for chunk in chunks
        ],
        chunk_file_times=[now for _ in paths],
        all_files=[Path("synthetic_notes") / path for path in paths],
        embeddings=np.array([[1.0, 0.0] for _ in paths]),
        chunk_embeddings=np.array([[1.0, 0.0] for _ in paths]),
        earliest_note=paths[-1] if paths else None,
        latest_note=paths[0] if paths else None,
    )


def _selectable_file_state(paths: list[str]) -> ConversationState:
    answer = "\n".join(
        ["当前知识库里的文件如下：", *[f"{i}. {path}" for i, path in enumerate(paths, 1)]]
    )
    return ConversationState(
        last_route="repo_meta",
        last_content_route="repo_meta",
        last_content_user_question="当前知识库有哪些文件？",
        last_answer_text=answer,
        last_answer_preview=answer,
        last_answer_type="enumeration_file",
        last_result_set_items=list(paths),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
    )


@pytest.mark.parametrize(
    "question",
    [
        "当前知识库有哪些文件？",
        "目前知识库里有什么文件？",
        "知识库中都有哪些文档？",
        "我的知识库里都有什么？",
        "现在有哪些文件？",
        "列出当前所有文件。",
    ],
)
def test_generic_file_lists_share_one_repo_wide_parse(question):
    assert parse_file_list_request(question) == ""
    assert extract_topic_from_list_request(question) == ""
    assert classify_repo_meta_question(question) == "list_files"
    assert detect_dialog_event(question, ConversationState(), _LoggerStub()).name == "repo_meta_request"


@pytest.mark.parametrize(
    ("question", "topic"),
    [
        ("成都有哪些文件？", "成都"),
        ("文档管理有哪些文件？", "文档管理"),
        ("文件系统有哪些文档？", "文件系统"),
        ("资料管理有哪些文件？", "资料管理"),
        ("当前项目有哪些文件？", "项目"),
        ("目前招聘相关有哪些资料？", "招聘"),
        ("现在这个合同有哪些文档？", "合同"),
        ("采购相关有哪些文件？", "采购"),
        ("关于合同的文件有哪些？", "合同"),
    ],
)
def test_topic_file_lists_keep_the_topic_core_intact(question, topic):
    assert parse_file_list_request(question) == topic
    assert extract_topic_from_list_request(question) == topic
    assert classify_repo_meta_question(question) == "list_files_by_topic"


@pytest.mark.parametrize(
    ("question", "classification"),
    [
        ("最近修改了哪些文件？", "time"),
        ("今天创建了哪些文档？", "time"),
        ("最早的文件有哪些？", "time"),
        ("最近修改的项目文件有哪些？", "time"),
        ("有多少个文件？", "count"),
        ("有哪些文件类型？", "list_files"),
        ("知识库有多大？", None),
    ],
)
def test_time_and_adjacent_repo_meta_contracts_stay_unchanged(question, classification):
    assert classify_repo_meta_question(question) == classification


def test_generic_file_list_creates_visible_ordered_set_and_empty_repo_does_not():
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    answer, topic = answer_repo_meta_question("当前知识库有哪些文件？", _repo_state(paths))
    state = update_state_after_local_answer(
        ConversationState(),
        question="当前知识库有哪些文件？",
        answer=answer,
        route="repo_meta",
        local_topic=topic,
        is_content_answer=True,
    )
    assert state.last_result_set_items == paths
    assert state.last_result_set_selectable is True
    assert [line for line in answer.splitlines() if line[:1].isdigit()] == [
        "1. 资料甲.md",
        "2. 记录乙.txt",
        "3. 说明丙.pdf",
    ]

    empty_answer, empty_topic = answer_repo_meta_question(
        "当前知识库有哪些文件？",
        _repo_state([]),
    )
    empty_state = update_state_after_local_answer(
        ConversationState(),
        question="当前知识库有哪些文件？",
        answer=empty_answer,
        route="repo_meta",
        local_topic=empty_topic,
        is_content_answer=True,
    )
    assert "还没有可用文档" in empty_answer
    assert empty_state.last_result_set_items is None
    assert resolve_file_result_set_selection("这些文件分别讲了什么？", []) is None


@pytest.mark.parametrize(
    "question",
    [
        "这些文件分别讲了什么？",
        "它们各自说了什么？",
        "对比一下这些文件。",
        "比较一下它们。",
    ],
)
def test_explicit_whole_set_reference_returns_visible_paths(question):
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    selection = resolve_file_result_set_selection(question, paths)
    assert selection is not None
    assert selection.paths == tuple(paths)
    assert selection.rejection is None


@pytest.mark.parametrize(
    "question",
    ["第二个文件讲了什么？", "第2条再展开。", "第二项详细说说。"],
)
def test_single_reference_selects_second_visible_file(question):
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    selection = resolve_file_result_set_selection(question, paths)
    assert selection is not None
    assert selection.paths == ("记录乙.txt",)
    event = detect_dialog_event(question, _selectable_file_state(paths), _LoggerStub())
    assert event.name == "result_set_followup"


@pytest.mark.parametrize("question", ["第四个文件讲了什么？", "第10条再展开。"])
def test_out_of_range_reference_returns_local_rejection(question):
    selection = resolve_file_result_set_selection(
        question,
        ["资料甲.md", "记录乙.txt", "说明丙.pdf"],
    )
    assert selection is not None
    assert selection.paths == ()
    assert selection.rejection == "当前结果集中只有 3 个文件，请选择第 1～3 个。"


@pytest.mark.parametrize(
    "question",
    [
        "前两个分别说说。",
        "第一个和第三个对比一下。",
        "除了第二个，其他文件分别讲了什么？",
    ],
)
def test_unsupported_complex_reference_returns_local_rejection(question):
    selection = resolve_file_result_set_selection(
        question,
        ["资料甲.md", "记录乙.txt", "说明丙.pdf"],
    )
    assert selection is not None
    assert selection.paths == ()
    assert "当前只支持选择单个文件或整个文件集合" in selection.rejection


@pytest.mark.parametrize(
    "question",
    [
        "比较 Python 和 Java。",
        "请把采购风险做个表。",
        "总结一下人工智能的发展。",
        "当前项目有哪些风险？",
        "第二章讲了什么？",
        "第二个问题如何解决？",
        "当前版本的第二个变化是什么？",
    ],
)
def test_independent_or_non_file_ordinal_question_has_no_file_scope(question):
    assert resolve_file_result_set_selection(
        question,
        ["资料甲.md", "记录乙.txt", "说明丙.pdf"],
    ) is None


def _run_turns(
    monkeypatch,
    tmp_path,
    *,
    questions: list[str],
    repo_paths: list[str],
    state: ConversationState,
):
    inputs = iter([*questions, "q"])
    allowed_paths: list[object] = []
    query_result_sets: list[object] = []
    real_materials = chat_runner.build_retrieval_materials
    real_query = chat_runner.build_search_query
    client = _ClientStub()

    monkeypatch.setattr(chat_runtime, "_read_user_question", lambda **_kwargs: next(inputs))
    monkeypatch.setattr(chat_runtime, "_flush_pending_tty_input_unix", lambda: False)
    monkeypatch.setattr(chat_runtime, "conversation_state", state)
    monkeypatch.setattr(
        chat_runner,
        "resolve_route",
        lambda current_question, *_args, **_kwargs: {
            "route": "normal_retrieval",
            "smalltalk_reply": "",
            "route_question_input": current_question,
        },
    )
    monkeypatch.setattr(
        "app.retrieval_flow.query.rewrite_search_query",
        lambda question, *_args, **_kwargs: question,
    )

    def capture_materials(**kwargs):
        allowed_paths.append(kwargs["allowed_paths"])
        return real_materials(**kwargs)

    def capture_query(**kwargs):
        query_result_sets.append(kwargs["last_result_set_items"])
        return real_query(**kwargs)

    monkeypatch.setattr(chat_runner, "build_retrieval_materials", capture_materials)
    monkeypatch.setattr(chat_runner, "build_search_query", capture_query)
    chat_runtime.run_chat_loop(
        _indexed_repo_state(repo_paths),
        _EmbeddingStub(),
        client,
        "offline-model",
        "http://127.0.0.1:9",
        "offline-model",
        _LoggerStub(),
        notes_dir=tmp_path / "notes",
        change_log_file=tmp_path / "changes.db",
        domain_dispatch_port=EmptyDomainHost(),
    )
    return allowed_paths, query_result_sets, client


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("这些文件分别讲了什么？", {"资料甲.md", "记录乙.txt", "说明丙.pdf"}),
        ("第2条再展开。", {"记录乙.txt"}),
    ],
)
def test_runner_uses_one_visible_scope_for_retrieval_and_query_context(
    monkeypatch,
    tmp_path,
    question,
    expected,
):
    visible = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    allowed, query_sets, _client = _run_turns(
        monkeypatch,
        tmp_path,
        questions=[question],
        repo_paths=[*visible, "隐藏丁.md"],
        state=_selectable_file_state(visible),
    )
    assert allowed == [expected]
    assert query_sets == [list(expected) if len(expected) == 1 else visible]
    assert "隐藏丁.md" not in allowed[0]


@pytest.mark.parametrize(
    "question",
    ["比较 Python 和 Java。", "请把采购风险做个表。", "总结一下人工智能的发展。"],
)
def test_runner_does_not_inherit_old_file_set_for_independent_question(
    monkeypatch,
    tmp_path,
    question,
):
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    allowed, query_sets, _client = _run_turns(
        monkeypatch,
        tmp_path,
        questions=[question],
        repo_paths=paths,
        state=_selectable_file_state(paths),
    )
    assert allowed == [None]
    assert query_sets == [None]


def test_runner_rejects_out_of_range_then_keeps_set_for_valid_choice(
    monkeypatch,
    tmp_path,
    capsys,
):
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    allowed, query_sets, client = _run_turns(
        monkeypatch,
        tmp_path,
        questions=["第四个文件讲了什么？", "第2条再展开。"],
        repo_paths=paths,
        state=_selectable_file_state(paths),
    )
    assert "当前结果集中只有 3 个文件，请选择第 1～3 个。" in capsys.readouterr().out
    assert allowed == [{"记录乙.txt"}]
    assert query_sets == [["记录乙.txt"]]
    assert chat_runtime.conversation_state.last_result_set_items == paths
    assert len(client.models.calls) <= 1


@pytest.mark.parametrize(
    "question",
    ["前两个分别说说。", "第一个和第三个对比一下。", "除了第二个，其他文件分别讲了什么？"],
)
def test_runner_rejects_unsupported_selection_without_retrieval_or_model(
    monkeypatch,
    tmp_path,
    question,
):
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    allowed, query_sets, client = _run_turns(
        monkeypatch,
        tmp_path,
        questions=[question],
        repo_paths=paths,
        state=_selectable_file_state(paths),
    )
    assert allowed == []
    assert query_sets == []
    assert client.models.calls == []
    assert chat_runtime.conversation_state.last_result_set_items == paths


def test_real_three_turn_cli_lists_then_scopes_then_selects_second(
    monkeypatch,
    tmp_path,
    capsys,
):
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    inputs = iter(
        ["当前知识库有哪些文件？", "这些文件分别讲了什么？", "第2条再展开。", "q"]
    )
    monkeypatch.setattr(chat_runtime, "_read_user_question", lambda **_kwargs: next(inputs))
    monkeypatch.setattr(chat_runtime, "_flush_pending_tty_input_unix", lambda: False)
    monkeypatch.setattr(chat_runtime, "conversation_state", ConversationState())
    monkeypatch.setattr(
        "app.retrieval_flow.query.rewrite_search_query",
        lambda question, *_args, **_kwargs: question,
    )
    chat_runtime.run_chat_loop(
        _indexed_repo_state(paths),
        _EmbeddingStub(),
        _ClientStub(),
        "offline-model",
        "http://127.0.0.1:9",
        "offline-model",
        _LoggerStub(),
        notes_dir=tmp_path / "notes",
        change_log_file=tmp_path / "changes.db",
        domain_dispatch_port=EmptyDomainHost(),
    )
    output = capsys.readouterr().out
    assert "1. 资料甲.md" in output
    assert "2. 记录乙.txt" in output
    assert "3. 说明丙.pdf" in output
    assert chat_runtime.conversation_state.last_result_set_items == paths
