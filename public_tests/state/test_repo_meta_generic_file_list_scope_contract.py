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
from app.chat_loop_handlers import try_handle_retrieval_force_local_or_empty_context
from app.chat_loop_handlers.guards import looks_like_analytic_retrieval_question
from app.chat_state_helpers import update_state_after_local_answer
from app.dialog.result_set import (
    materialize_single_file_result_set_question,
    resolve_file_result_set_selection,
)
from app.dialog.state_machine import ConversationState, detect_dialog_event
from app.domain_host.host import EmptyDomainHost
from app.file_actions.request_mutations import handle_rename_request_action
from retrieval.repo_index_types import RepoState


class _LoggerStub:
    def debug(self, *_args, **_kwargs):
        return None

    info = debug
    warning = debug
    error = debug


class _RecordingLogger(_LoggerStub):
    def __init__(self):
        self.messages: list[str] = []

    def debug(self, message, *_args, **_kwargs):
        self.messages.append(str(message))

    info = debug
    warning = debug
    error = debug


class _EmbeddingStub:
    def encode(self, texts):
        return np.array([[0.8, 0.0] for _ in texts])


@pytest.mark.parametrize(
    ("question", "selected_path", "expected"),
    [
        (
            "第二个文件再详细说说。",
            "/synthetic/notes/系统设计.md",
            "文件《系统设计.md》再详细说说。",
        ),
        (
            "那第十四个文件呢？它最看重什么？",
            r"C:\synthetic\notes\API文档.md",
            "文件《API文档.md》呢？它最看重什么？",
        ),
    ],
)
def test_materialized_result_set_question_uses_display_name_and_keeps_intent(
    question,
    selected_path,
    expected,
):
    materialized = materialize_single_file_result_set_question(question, selected_path)

    assert materialized == expected
    assert "/synthetic/notes" not in materialized
    assert r"C:\synthetic\notes" not in materialized


class _ModelsStub:
    def __init__(self):
        self.calls: list[str] = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(contents)
        return SimpleNamespace(text="资料甲记录检索，记录乙说明路由，说明丙描述归档。")


class _ClientStub:
    def __init__(self):
        self.models = _ModelsStub()


class _DetailModelsStub(_ModelsStub):
    def generate_content(self, *, model, contents, config=None):
        self.calls.append(contents)
        return SimpleNamespace(text="根据当前受限片段，已展开说明其中的合成知识内容。")


class _DetailClientStub:
    def __init__(self):
        self.models = _DetailModelsStub()


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


def _indexed_repo_state(paths: list[str], chunks: list[str] | None = None) -> RepoState:
    now = datetime(2026, 7, 24, 12, 0, 0)
    chunks = list(chunks) if chunks is not None else [f"{path} 的合成知识内容。" for path in paths]
    assert len(chunks) == len(paths)
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


def _focused_file_state(paths: list[str], focus_file: str) -> ConversationState:
    state = _selectable_file_state(paths)
    state.last_route = "normal_retrieval"
    state.last_content_route = "normal_retrieval"
    state.last_content_user_question = "请详细说明当前文件。"
    state.last_effective_search_query = "当前文件"
    state.last_result_set_focus_file = focus_file
    return state


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
    repo_chunks: list[str] | None = None,
):
    inputs = iter([*questions, "q"])
    allowed_paths: list[object] = []
    query_result_sets: list[object] = []
    real_materials = chat_runner.build_retrieval_materials
    real_query = chat_runner.build_search_query
    client = _ClientStub()
    logger = _LoggerStub()

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
        _indexed_repo_state(repo_paths, repo_chunks),
        _EmbeddingStub(),
        client,
        "offline-model",
        "http://127.0.0.1:9",
        "offline-model",
        logger,
        notes_dir=tmp_path / "notes",
        change_log_file=tmp_path / "changes.db",
        domain_dispatch_port=EmptyDomainHost(),
    )
    return allowed_paths, query_result_sets, client


class _AcceptanceModelsStub(_ModelsStub):
    def __init__(self):
        super().__init__()
        self.latest_questions: list[str] = []

    def generate_content(self, *, model, contents, config=None):
        self.calls.append(contents)
        latest_block = contents.split("【用户最新提问】", 1)[-1].strip()
        latest_question = latest_block.splitlines()[0].strip()
        self.latest_questions.append(latest_question)
        if "客户项目交付" in latest_question:
            answer = "第五项更偏客户项目交付；第五项强调交付，第二项更偏内部平台。"
        elif "最看重什么" in latest_question:
            answer = (
                "第十四项最看重以下能力：\n"
                "1. 完整交付能力。\n"
                "2. 跨团队协作。\n"
                "3. 问题闭环。"
            )
        elif "文件《合成岗位资料10.png》再详细说说" in latest_question:
            answer = "第十项主要说明项目协调、交付流程和质量保障。"
        elif "第十个文件" in latest_question:
            answer = "目前无法确定哪个文件是第十个。"
        else:
            answer = "根据当前受限片段完成内容回答。"
        return SimpleNamespace(text=answer)


class _AcceptanceClientStub:
    def __init__(self):
        self.models = _AcceptanceModelsStub()


def _run_complete_result_set_acceptance_chain(monkeypatch, tmp_path):
    paths = [f"合成岗位资料{i:02d}.png" for i in range(1, 15)]
    chunks = [f"第{i}份合成资料描述通用职责和能力。" for i in range(1, 15)]
    chunks[1] = "该资料侧重内部平台研发、技术架构和工程效率。"
    chunks[4] = "该资料侧重客户项目实施、现场交付和需求沟通。"
    chunks[9] = "该资料说明项目协调、交付流程和质量保障。"
    chunks[13] = "该资料最看重完整交付能力、跨团队协作和问题闭环。"
    questions = [
        "当前知识库有哪些文件？",
        "这些文件分别讲了什么？",
        "第二个文件再详细说说。",
        "这个岗位对学历、开发年限和AI项目经验分别有什么要求？",
        "这些要求里，哪些是硬门槛，哪些更像优先项？",
        "第五个文件也详细说说。",
        "它和第二个文件相比，哪个更偏客户项目交付？",
        "那第十四个文件呢？它最看重什么？",
        "第二十个文件再详细说说。",
        "那就改成第十个文件。",
        "Python里的生成器和普通函数有什么区别？",
        "月球为什么总是同一面朝向地球？",
    ]
    inputs = iter([*questions, "q"])
    client = _AcceptanceClientStub()
    logger = _RecordingLogger()
    material_calls: list[dict] = []
    query_calls: list[dict] = []
    direct_answers: list[str] = []
    locator_answers: list[str] = []
    state_snapshots: list[dict] = []
    real_materials = chat_runner.build_retrieval_materials
    real_query = chat_runner.build_search_query
    real_direct = chat_runner.maybe_build_direct_lookup_answer
    real_locator = chat_runner.maybe_build_file_location_answer
    real_state_update = chat_runner.update_state_after_retrieval_answer

    monkeypatch.setattr(chat_runtime, "_read_user_question", lambda **_kwargs: next(inputs))
    monkeypatch.setattr(chat_runtime, "_flush_pending_tty_input_unix", lambda: False)
    monkeypatch.setattr(chat_runtime, "conversation_state", ConversationState())
    monkeypatch.setattr(chat_runner, "print_answer", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        chat_runner,
        "resolve_route",
        lambda current_question, *_args, **_kwargs: {
            "route": (
                "repo_meta"
                if current_question == "当前知识库有哪些文件？"
                else "normal_retrieval"
            ),
            "smalltalk_reply": "",
            "route_question_input": current_question,
        },
    )
    monkeypatch.setattr(
        "app.retrieval_flow.query.rewrite_search_query",
        lambda question, *_args, **_kwargs: question,
    )

    def capture_materials(**kwargs):
        material_calls.append(
            {
                "question": kwargs["question"],
                "allowed_paths": kwargs["allowed_paths"],
            }
        )
        return real_materials(**kwargs)

    def capture_query(**kwargs):
        query_calls.append(
            {
                "question": kwargs["question"],
                "items": kwargs["last_result_set_items"],
                "entity": kwargs["last_result_set_entity_type"],
            }
        )
        return real_query(**kwargs)

    def capture_direct(**kwargs):
        answer = real_direct(**kwargs)
        if answer:
            direct_answers.append(kwargs["question"])
        return answer

    def capture_locator(**kwargs):
        answer = real_locator(**kwargs)
        if answer:
            locator_answers.append(kwargs["question"])
        return answer

    def capture_state_update(state, question, answer_text, current_logger, **kwargs):
        updated = real_state_update(
            state,
            question,
            answer_text,
            current_logger,
            **kwargs,
        )
        state_snapshots.append(
            {
                "question": question,
                "answer": answer_text,
                "answer_type": updated.last_answer_type,
                "focus": updated.last_result_set_focus_file,
                "items": list(updated.last_result_set_items or []),
                "entity": updated.last_result_set_entity_type,
            }
        )
        return updated

    monkeypatch.setattr(chat_runner, "build_retrieval_materials", capture_materials)
    monkeypatch.setattr(chat_runner, "build_search_query", capture_query)
    monkeypatch.setattr(chat_runner, "maybe_build_direct_lookup_answer", capture_direct)
    monkeypatch.setattr(chat_runner, "maybe_build_file_location_answer", capture_locator)
    monkeypatch.setattr(chat_runner, "update_state_after_retrieval_answer", capture_state_update)
    chat_runtime.run_chat_loop(
        _indexed_repo_state(paths, chunks),
        _EmbeddingStub(),
        client,
        "offline-model",
        "http://127.0.0.1:9",
        "offline-model",
        logger,
        notes_dir=tmp_path / "notes",
        change_log_file=tmp_path / "changes.db",
        domain_dispatch_port=EmptyDomainHost(),
    )
    return SimpleNamespace(
        paths=paths,
        material_calls=material_calls,
        query_calls=query_calls,
        direct_answers=direct_answers,
        locator_answers=locator_answers,
        state_snapshots=state_snapshots,
        model_calls=client.models.calls,
        model_questions=client.models.latest_questions,
        final_state=chat_runtime.conversation_state,
        logs=logger.messages,
    )


@pytest.mark.parametrize(
    "contract",
    [
        "comparison_generation",
        "single_detail_materialization",
        "ordinal_content_generation",
        "correction_semantics",
        "independent_exit",
    ],
)
def test_complete_result_set_acceptance_chain(monkeypatch, tmp_path, contract):
    result = _run_complete_result_set_acceptance_chain(monkeypatch, tmp_path)

    if contract == "comparison_generation":
        question = "它和第二个文件相比，哪个更偏客户项目交付？"
        call = next(item for item in result.material_calls if item["question"] == question)
        assert call["allowed_paths"] == {result.paths[4], result.paths[1]}
        assert question not in result.direct_answers
        assert question in result.model_questions
        snapshot = next(item for item in result.state_snapshots if item["question"] == question)
        assert "第五项更偏客户项目交付" in snapshot["answer"]
        assert snapshot["answer_type"] is None
        assert snapshot["focus"] == result.paths[4]
        assert snapshot["items"] == result.paths
    elif contract == "single_detail_materialization":
        generation_question = next(
            question
            for question in result.model_questions
            if "合成岗位资料02.png" in question
        )
        assert generation_question == "文件《合成岗位资料02.png》再详细说说。"
        assert "第二个文件" not in generation_question
    elif contract == "ordinal_content_generation":
        question = "那第十四个文件呢？它最看重什么？"
        call = next(item for item in result.material_calls if item["question"] == question)
        assert call["allowed_paths"] == {result.paths[13]}
        assert question not in result.locator_answers
        assert question not in result.direct_answers
        generation_question = next(
            item
            for item in result.model_questions
            if "合成岗位资料14.png" in item
        )
        assert "第十四个文件" not in generation_question
        assert "最看重什么" in generation_question
        snapshot = next(item for item in result.state_snapshots if item["question"] == question)
        assert snapshot["answer_type"] is None
        assert snapshot["focus"] == result.paths[13]
        assert snapshot["items"] == result.paths
    elif contract == "correction_semantics":
        corrected_question = "第十个文件再详细说说。"
        call = next(
            item for item in result.material_calls
            if item["allowed_paths"] == {result.paths[9]}
        )
        assert call["question"] == corrected_question
        generation_question = next(
            question
            for question in result.model_questions
            if "合成岗位资料10.png" in question
        )
        assert generation_question == "文件《合成岗位资料10.png》再详细说说。"
        assert "第十个文件" not in generation_question
        assert not any(
            item["question"] == "第二十个文件再详细说说。"
            for item in result.material_calls
        )
        assert "第二十个文件再详细说说。" not in result.model_questions
        assert corrected_question not in result.model_questions
        assert "那就改成第十个文件。" not in result.model_questions
        snapshot = next(
            item for item in result.state_snapshots
            if item["question"] == corrected_question
        )
        assert "无法确定" not in snapshot["answer"]
        assert "无法切换" not in snapshot["answer"]
        assert snapshot["focus"] == result.paths[9]
        assert snapshot["items"] == result.paths
    else:
        independent_questions = {
            "Python里的生成器和普通函数有什么区别？",
            "月球为什么总是同一面朝向地球？",
        }
        calls = {
            item["question"]: item["allowed_paths"]
            for item in result.material_calls
            if item["question"] in independent_questions
        }
        assert calls == {question: None for question in independent_questions}
        query_contexts = {
            item["question"]: (item["items"], item["entity"])
            for item in result.query_calls
            if item["question"] in independent_questions
        }
        assert query_contexts == {
            question: (None, None)
            for question in independent_questions
        }
        assert result.final_state.last_result_set_focus_file is None
        assert result.final_state.last_result_set_items is None
        assert result.final_state.last_result_set_entity_type is None


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


def test_runner_and_retrieval_state_writeback_share_question_scope_instances(
    monkeypatch,
    tmp_path,
):
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    built_signals = []
    built_decisions = []
    consumed_facts = []
    real_analyze = chat_runner.analyze_question_signals
    real_decide = chat_runner.decide_file_result_set_scope
    real_update = chat_runner.update_state_after_retrieval_answer

    def capture_analyze(*args, **kwargs):
        signals = real_analyze(*args, **kwargs)
        built_signals.append(signals)
        return signals

    def capture_decide(*args, **kwargs):
        assert kwargs["signals"] is built_signals[-1]
        decision = real_decide(*args, **kwargs)
        built_decisions.append(decision)
        return decision

    def capture_update(*args, **kwargs):
        consumed_facts.append(
            (kwargs["question_signals"], kwargs["scope_decision"])
        )
        return real_update(*args, **kwargs)

    monkeypatch.setattr(chat_runner, "analyze_question_signals", capture_analyze)
    monkeypatch.setattr(chat_runner, "decide_file_result_set_scope", capture_decide)
    monkeypatch.setattr(chat_runner, "update_state_after_retrieval_answer", capture_update)

    _run_turns(
        monkeypatch,
        tmp_path,
        questions=["第二个文件再详细说说。"],
        repo_paths=paths,
        state=_selectable_file_state(paths),
    )

    assert len(built_signals) == 1
    assert len(built_decisions) == 1
    assert consumed_facts == [(built_signals[0], built_decisions[0])]
    assert consumed_facts[0][0] is built_signals[0]
    assert consumed_facts[0][1] is built_decisions[0]


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


def test_result_set_focus_file_should_survive_single_item_expansion():
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    state = ConversationState(
        last_route="normal_retrieval",
        last_content_route="normal_retrieval",
        last_content_user_question="第二个文件再详细说说。",
        last_answer_text="根据当前受限片段，已展开说明其中的合成知识内容。",
        last_answer_preview="根据当前受限片段，已展开说明其中的合成知识内容。",
        last_answer_type=None,
        last_result_set_items=list(paths),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
        last_result_set_focus_file="记录乙.txt",
    )

    event = detect_dialog_event(
        "这个岗位对学历、开发年限和AI项目经验分别有什么要求？",
        state,
        _LoggerStub(),
        focused_file="记录乙.txt",
    )

    assert event.name == "content_followup"
    assert state.last_result_set_focus_file == "记录乙.txt"
    assert state.last_result_set_items == paths


@pytest.mark.parametrize(
    "question",
    [
        "它主要解决什么问题？",
        "这个方案为什么这么设计？",
        "其中有哪些风险？",
        "这个结论是怎么得出的？",
        "这里有没有矛盾？",
    ],
)
def test_generic_explicit_focus_reference_is_content_followup(question):
    paths = ["系统设计.md", "旅行记录.txt", "实验报告.pdf"]
    state = _focused_file_state(paths, "系统设计.md")

    event = detect_dialog_event(
        question,
        state,
        _LoggerStub(),
        focused_file="系统设计.md",
    )

    assert event.name == "content_followup"


@pytest.mark.parametrize(
    "question",
    [
        "Python 开发经验一般怎么衡量？",
        "学历认证是什么意思？",
        "这个项目的重点应该如何确定？",
        "工作年限通常怎样计算？",
    ],
)
def test_domain_terms_do_not_bind_independent_question_to_old_focus(question):
    paths = ["系统设计.md", "旅行记录.txt", "实验报告.pdf"]
    state = _focused_file_state(paths, "系统设计.md")

    event = detect_dialog_event(
        question,
        state,
        _LoggerStub(),
        focused_file="系统设计.md",
    )

    assert event.name != "content_followup"
    assert event.name != "result_set_followup"


@pytest.mark.parametrize(
    ("question", "expected_index"),
    [
        ("第六个文件是谁写的？", 6),
        ("第三个文件是什么时候创建的？", 3),
        ("第九份资料是否提到了某个接口？", 9),
        ("第十二个文件适合什么场景？", 12),
        ("第十个文件靠谱吗？", 10),
        ("那第七个呢？", 7),
    ],
)
def test_explicit_ordinal_reference_selects_without_content_marker(
    question,
    expected_index,
):
    paths = [f"合成资料{i:02d}.md" for i in range(1, 15)]

    selection = resolve_file_result_set_selection(question, paths)

    assert selection is not None
    assert selection.paths == (paths[expected_index - 1],)
    event = detect_dialog_event(question, _selectable_file_state(paths), _LoggerStub())
    assert event.name == "result_set_followup"


@pytest.mark.parametrize(
    "question",
    [
        "它主要解决什么问题？",
        "其中有哪些风险？",
        "这个结论是怎么得出的？",
        "这里有没有矛盾？",
    ],
)
def test_runner_scopes_generic_focus_reference_to_current_file(
    monkeypatch,
    tmp_path,
    question,
):
    paths = ["系统设计.md", "旅行记录.txt", "实验报告.pdf", "API文档.md"]
    allowed, query_sets, _client = _run_turns(
        monkeypatch,
        tmp_path,
        questions=[question],
        repo_paths=paths,
        state=_focused_file_state(paths, "系统设计.md"),
        repo_chunks=[
            "系统设计方案说明组件边界与风险。",
            "旅行记录描述公开景点。",
            "实验报告记录合成实验。",
            "API 文档说明接口调用。",
        ],
    )

    assert allowed == [{"系统设计.md"}]
    assert query_sets == [["系统设计.md"]]


@pytest.mark.parametrize(
    "question",
    [
        "Python 开发经验一般怎么衡量？",
        "学历认证是什么意思？",
        "这个项目的重点应该如何确定？",
        "工作年限通常怎样计算？",
    ],
)
def test_runner_exits_old_focus_for_standalone_general_question(
    monkeypatch,
    tmp_path,
    question,
):
    paths = ["系统设计.md", "旅行记录.txt", "实验报告.pdf"]
    allowed, query_sets, _client = _run_turns(
        monkeypatch,
        tmp_path,
        questions=[question],
        repo_paths=paths,
        state=_focused_file_state(paths, "系统设计.md"),
    )

    assert allowed == [None]
    assert query_sets == [None]
    assert chat_runtime.conversation_state.last_result_set_focus_file is None


def test_runner_keeps_original_focus_contract_for_referential_followups(
    monkeypatch,
    tmp_path,
):
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    allowed, query_sets, _client = _run_turns(
        monkeypatch,
        tmp_path,
        questions=[
            "这个岗位对学历、开发年限和AI项目经验分别有什么要求？",
            "这些要求里，哪些是硬门槛，哪些更像优先项？",
        ],
        repo_paths=paths,
        state=_focused_file_state(paths, "记录乙.txt"),
    )

    assert allowed == [{"记录乙.txt"}, {"记录乙.txt"}]
    assert query_sets == [["记录乙.txt"], ["记录乙.txt"]]
    assert chat_runtime.conversation_state.last_result_set_items == paths
    assert chat_runtime.conversation_state.last_result_set_focus_file == "记录乙.txt"


def test_runner_preserves_ordered_set_across_switch_compare_and_correction(
    monkeypatch,
    tmp_path,
    capsys,
):
    paths = [f"合成资料{i:02d}.md" for i in range(1, 15)]
    allowed, query_sets, _client = _run_turns(
        monkeypatch,
        tmp_path,
        questions=[
            "第五个文件也详细说说。",
            "它和第二个文件相比，哪个更适合当前目标？",
            "那第十四个文件呢？它最看重什么？",
            "第二十个文件再详细说说。",
            "那就改成第十个文件。",
        ],
        repo_paths=paths,
        state=_focused_file_state(paths, paths[1]),
    )

    assert "当前结果集中只有 14 个文件，请选择第 1～14 个。" in capsys.readouterr().out
    assert allowed == [
        {paths[4]},
        {paths[4], paths[1]},
        {paths[13]},
        {paths[9]},
    ]
    assert query_sets == [
        [paths[4]],
        [paths[4], paths[1]],
        [paths[13]],
        [paths[9]],
    ]
    assert chat_runtime.conversation_state.last_result_set_items == paths
    assert chat_runtime.conversation_state.last_result_set_focus_file == paths[9]


def test_real_rename_request_still_enters_rename_preview(tmp_path):
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    source_name = "旧名称.txt"
    (notes_dir / source_name).write_text("合成测试内容", encoding="utf-8")
    state = ConversationState()

    handled, state, _focus_file = handle_rename_request_action(
        question="把“旧名称.txt”改成“新名称.txt”",
        start_qa=0.0,
        state=state,
        memory_buffer=[],
        current_focus_file=None,
        repo_state=SimpleNamespace(paths=[source_name], docs=["合成测试内容"], doc_records=[]),
        repo_paths=[source_name],
        notes_dir=notes_dir,
    )

    assert handled is True
    assert state.pending_action_type == "rename"
    assert state.pending_action_source_path == source_name
    assert state.pending_action_target_path == "新名称.txt"


def test_result_set_comparison_should_bind_current_focus_and_ordinal():
    selection = resolve_file_result_set_selection(
        "它和第二个文件相比，哪个更偏客户项目交付？",
        ["资料甲.md", "记录乙.txt", "说明丙.pdf"],
        focus_file="资料甲.md",
    )

    assert selection is not None
    assert selection.paths == ("资料甲.md", "记录乙.txt")


@pytest.mark.parametrize(
    "question",
    [
        "第二个文件再详细说说。",
        "第二个文件展开讲讲。",
        "第二个文件具体分析。",
        "第二个文件深入说明。",
    ],
)
def test_selected_file_expansion_uses_generation_without_expanding_scope(
    monkeypatch,
    tmp_path,
    question,
):
    visible = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    assert looks_like_analytic_retrieval_question(question) is True

    allowed, query_sets, client = _run_turns(
        monkeypatch,
        tmp_path,
        questions=[question],
        repo_paths=[*visible, "隐藏丁.md"],
        state=_selectable_file_state(visible),
    )

    assert allowed == [{"记录乙.txt"}]
    assert query_sets == [["记录乙.txt"]]
    assert len(client.models.calls) == 1
    assert "记录乙.txt" in client.models.calls[0]
    assert "隐藏丁.md" not in client.models.calls[0]
    assert chat_runtime.conversation_state.last_answer_type is None
    assert chat_runtime.conversation_state.last_result_set_items == visible


def test_failed_simple_local_fallback_does_not_consume_generation_turn():
    repo_state = SimpleNamespace(
        paths=["材料甲.md"],
        chunk_paths=["材料甲.md"],
        chunk_texts=["材料甲包含可供生成回答的背景段落。"],
    )

    answer = try_handle_retrieval_force_local_or_empty_context(
        route="normal_retrieval",
        question="它的负责人是谁？",
        event_name="result_set_followup",
        search_query="负责人",
        relevant_indices=[0],
        repo_state=repo_state,
        materials={"context_text": "【参考片段】材料甲.md | 背景段落", "inventory_candidates_text": ""},
        logger=_LoggerStub(),
    )

    assert answer is None


def test_successful_simple_lookup_stays_local_without_generation(
    monkeypatch,
    tmp_path,
):
    paths = ["资料甲.md", "记录乙.txt"]
    allowed, _query_sets, client = _run_turns(
        monkeypatch,
        tmp_path,
        questions=["哪份文件提到了 RAG？"],
        repo_paths=paths,
        state=ConversationState(),
        repo_chunks=["资料甲使用 RAG 检索。", "记录乙包含其他说明。"],
    )

    assert allowed == [None]
    assert client.models.calls == []


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


def test_real_three_turn_cli_lists_then_scopes_second_file_for_generation(
    monkeypatch,
    tmp_path,
    capsys,
):
    paths = ["资料甲.md", "记录乙.txt", "说明丙.pdf"]
    inputs = iter(
        ["当前知识库有哪些文件？", "这些文件分别讲了什么？", "第二个文件再详细说说。", "q"]
    )
    monkeypatch.setattr(chat_runtime, "_read_user_question", lambda **_kwargs: next(inputs))
    monkeypatch.setattr(chat_runtime, "_flush_pending_tty_input_unix", lambda: False)
    monkeypatch.setattr(chat_runtime, "conversation_state", ConversationState())
    monkeypatch.setattr(
        "app.retrieval_flow.query.rewrite_search_query",
        lambda question, *_args, **_kwargs: question,
    )
    allowed_paths: list[object] = []
    real_materials = chat_runner.build_retrieval_materials

    def capture_materials(**kwargs):
        allowed_paths.append(kwargs["allowed_paths"])
        return real_materials(**kwargs)

    monkeypatch.setattr(chat_runner, "build_retrieval_materials", capture_materials)
    client = _DetailClientStub()
    logger = _RecordingLogger()
    chat_runtime.run_chat_loop(
        _indexed_repo_state(paths),
        _EmbeddingStub(),
        client,
        "offline-model",
        "http://127.0.0.1:9",
        "offline-model",
        logger,
        notes_dir=tmp_path / "notes",
        change_log_file=tmp_path / "changes.db",
        domain_dispatch_port=EmptyDomainHost(),
    )
    output = capsys.readouterr().out
    assert "1. 资料甲.md" in output
    assert "2. 记录乙.txt" in output
    assert "3. 说明丙.pdf" in output
    assert "根据当前受限片段，已展开说明其中的合成知识内容。" in output
    assert allowed_paths == [set(paths), {"记录乙.txt"}]
    assert len(client.models.calls) == 2
    detail_prompt = client.models.calls[-1]
    detail_evidence = detail_prompt.split("【参考片段】:", 1)[1].split("【用户最新提问】", 1)[0]
    assert "记录乙.txt" in detail_evidence
    assert "资料甲.md" not in detail_evidence
    assert "说明丙.pdf" not in detail_evidence
    assert any("[文件结果集范围] 限定为 1 个文件" in message for message in logger.messages)
    assert any("[远程模型生成] 进入生成阶段" in message for message in logger.messages)
    assert chat_runtime.conversation_state.last_answer_type is None
    assert chat_runtime.conversation_state.last_result_set_items == paths
