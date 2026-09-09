from __future__ import annotations

import pytest

from ai.table_presentation import StructuredTable
from app import chat_loop as chat_runtime
import app.chat_loop_parts.runner as chat_runner
from app.chat_state_helpers import (
    update_state_after_answer_presentation,
    update_state_after_local_answer,
)
from app.dialog.state_machine import ConversationState
from app.domain_host.host import EmptyDomainHost
from public_tests.state.test_ordinal_target_retrieval_binding_contract import _scope
from public_tests.state.test_repo_meta_generic_file_list_scope_contract import (
    _DetailClientStub,
    _EmbeddingStub,
    _RecordingLogger,
    _indexed_repo_state,
)


@pytest.fixture(autouse=True)
def _isolate_relation_review(monkeypatch):
    # This module tests existing routing/presentation with a controlled model.
    # The separate relation-review contracts exercise the real review boundary.
    from ai.evidence_scope_review import EvidenceScopeReview
    from app.chat_loop_parts import runner

    monkeypatch.setattr(runner, "review_generated_evidence_scope", lambda **_: EvidenceScopeReview("VERIFIED"))


@pytest.mark.parametrize(
    "target",
    ["DOC-02_β采购说明.md", "DOC-02_café合同记录.md", "DOC-02_🧩课程资料.md"],
)
@pytest.mark.parametrize("reordered", [False, True])
def test_local_inventory_identity_survives_ordinal_focus_and_comparison(
    monkeypatch, tmp_path, target, reordered,
):
    paths = ["DOC-01_资料甲.md", target, "DOC-03_说明丙.md"]
    repo = _indexed_repo_state(paths)
    client = _DetailClientStub()
    snapshots = []
    scopes = []
    materials = []
    questions = iter([
        "当前知识库有哪些文件？",
        "看下第1个文件。" if reordered else "看下第2个文件。",
        "它主要讲了什么？",
        "把这个文件和第3个文件比较：各自讲了什么？",
        "q",
    ])

    def read_question(**_kwargs):
        if chat_runtime.conversation_state.last_answer_text:
            state = chat_runtime.conversation_state
            snapshots.append((list(state.last_result_set_items or []), state.last_result_set_focus_file))
            if len(snapshots) == 1:
                assert client.models.calls == []  # Inventory must remain local.
                assert snapshots[-1][0] == paths
                if reordered:
                    visible = [target, paths[0], paths[2]]
                    update_state_after_answer_presentation(
                        state, "整理成表格", "本地表格",
                        table=StructuredTable(
                            columns=("文件",), rows=tuple((p,) for p in visible),
                        ),
                    )
                    assert state.current_presentation_table.row_identities == tuple(visible)
                    assert _scope("第一份讲了什么？", state)[2].result_scope_paths == (target,)
        return next(questions)

    real_materials = chat_runner.build_retrieval_materials

    def capture_materials(**kwargs):
        scopes.append(kwargs["allowed_paths"])
        result = real_materials(**kwargs)
        materials.append(result)
        return result

    monkeypatch.setattr(chat_runtime, "_read_user_question", read_question)
    monkeypatch.setattr(chat_runtime, "_flush_pending_tty_input_unix", lambda: False)
    monkeypatch.setattr(chat_runtime, "conversation_state", ConversationState())
    monkeypatch.setattr(chat_runner, "build_retrieval_materials", capture_materials)
    monkeypatch.setattr(
        "app.retrieval_flow.query.rewrite_search_query",
        lambda question, *_args, **_kwargs: question,
    )
    chat_runtime.run_chat_loop(
        repo, _EmbeddingStub(), client, "offline-model", "http://127.0.0.1:9",
        "offline-model", _RecordingLogger(), notes_dir=tmp_path / "notes",
        change_log_file=tmp_path / "changes.db", domain_dispatch_port=EmptyDomainHost(),
    )

    assert scopes == [{target}, {target}, {target, paths[2]}]
    for result in materials[:2]:
        assert result["current_focus_file"] == target
        assert [repo.chunk_paths[i] for i in result["relevant_indices"]] == [target]
        assert target in result["context_text"]
        assert paths[0] not in result["context_text"]
        assert paths[2] not in result["context_text"]
    assert {repo.chunk_paths[i] for i in materials[2]["relevant_indices"]} == {target, paths[2]}
    assert all(items == paths for items, _focus in snapshots)
    assert all(focus == target for _items, focus in snapshots[1:])
    assert target in client.models.calls[0].split("【用户最新提问】", 1)[1]


@pytest.mark.parametrize("topic", ["list_files", "list_files_by_topic"])
@pytest.mark.parametrize(
    "target",
    ["DOC-02_ascii.md", "DOC-02_β.md", "目录/cafe\u0301_合同.md", "目录/资料  甲.md"],
)
def test_inventory_binding_preserves_exact_identity_and_visible_subset_order(topic, target):
    paths = ["目录甲/共享.md", target, "目录乙/共享.md"]
    visible = [paths[2], target]
    answer = "\n".join(f"{i}. {path}" for i, path in enumerate(visible, 1))
    if topic == "list_files_by_topic":
        answer = "\n".join(f"- {path}" for path in visible)
    state = update_state_after_local_answer(
        ConversationState(), "列出文件", answer, "repo_meta", topic, True,
        canonical_file_paths=paths,
    )

    assert state.last_result_set_items == visible
    assert state.last_result_set_selectable is True
    assert _scope("第一个文件讲了什么？", state)[2].result_scope_paths == (paths[2],)
    assert _scope("第二个文件讲了什么？", state)[2].result_scope_paths == (target,)


@pytest.mark.parametrize(
    "visible",
    [
        ["采购说明.md", "说明丙.md"],  # Truncated prefix cannot be repaired by guessing.
        ["DOC-02_β采购说明.md", "不存在.md"],
        ["DOC-02_β采购说明.md", "DOC-02_β采购说明.md"],
        ["共享.md", "说明丙.md"],  # Ambiguous basename is not canonical authority.
    ],
)
def test_unmapped_inventory_ordinal_fails_locally_without_retrieval_or_generation(
    monkeypatch, tmp_path, visible,
):
    paths = ["DOC-02_β采购说明.md", "目录甲/共享.md", "目录乙/共享.md", "说明丙.md"]
    answer = "\n".join(f"{i}. {path}" for i, path in enumerate(visible, 1))
    inputs = iter(["当前知识库有哪些文件？", "看下第2个文件。", "q"])
    client = _DetailClientStub()
    monkeypatch.setattr(chat_runtime, "_read_user_question", lambda **_kwargs: next(inputs))
    monkeypatch.setattr(chat_runtime, "_flush_pending_tty_input_unix", lambda: False)
    monkeypatch.setattr(chat_runtime, "conversation_state", ConversationState())
    monkeypatch.setattr(chat_runtime, "try_handle_repo_meta", lambda *_args, **_kwargs: (answer, "list_files"))

    def forbidden_retrieval(**_kwargs):
        pytest.fail("An unmapped inventory ordinal must be rejected before retrieval")

    monkeypatch.setattr(chat_runner, "build_retrieval_materials", forbidden_retrieval)
    chat_runtime.run_chat_loop(
        _indexed_repo_state(paths), _EmbeddingStub(), client, "offline-model",
        "http://127.0.0.1:9", "offline-model", _RecordingLogger(),
        notes_dir=tmp_path / "notes", change_log_file=tmp_path / "changes.db",
        domain_dispatch_port=EmptyDomainHost(),
    )
    assert client.models.calls == []
    assert "无法可靠确定" in chat_runtime.conversation_state.last_answer_text
    assert chat_runtime.conversation_state.last_result_set_selectable is False
