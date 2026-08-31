from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from ai.repo_meta.classifier import classify_repo_meta_question
from app import chat_loop as chat_runtime
import app.chat_loop_parts.runner as chat_runner
import app.dialog.result_set as result_set_operations
from app.dialog.result_set import resolve_file_result_set_selection
from app.dialog.state_machine import ConversationState


class _LoggerStub:
    def debug(self, *_args, **_kwargs):
        return None

    info = debug
    warning = debug
    error = debug


class _ModelsSpy:
    def __init__(self, calls):
        self._calls = calls

    def generate_content(self, **_kwargs):
        self._calls["remote_model"] += 1
        raise AssertionError("详情操作不应调用远程模型")


class _ClientSpy:
    def __init__(self, calls):
        self.models = _ModelsSpy(calls)


def _write_synthetic_file(path: Path, size: int, timestamp: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (timestamp, timestamp))


def _repo_state(
    notes_dir: Path,
    paths: list[str],
    *,
    sizes: dict[str, int] | None = None,
    times: dict[str, datetime] | None = None,
):
    sizes = sizes or {}
    times = times or {}
    normalized_paths = [path.replace("\\", "/") for path in paths]
    return SimpleNamespace(
        paths=list(paths),
        all_files=[notes_dir / path for path in normalized_paths],
        file_times=[times.get(path) for path in paths],
        file_info_list=[],
        doc_records=[
            {
                "path": path,
                "file_size": sizes.get(path),
                "file_time": times.get(path),
            }
            for path in paths
        ],
        docs=["禁止读取的合成正文" for _ in paths],
        earliest_note="合成最早记录",
        latest_note="合成最新记录",
    )


def _active_state(items: list[str], *, local_topic: str | None = "list_files"):
    return ConversationState(
        last_user_question="按文件名排列",
        last_route="repo_meta" if local_topic else "normal_retrieval",
        last_local_topic=local_topic,
        last_answer_text="合成文件列表",
        last_answer_preview="合成文件列表",
        last_answer_type="enumeration_file",
        last_content_route="normal_retrieval",
        last_content_user_question="此前的合成内容问题",
        last_content_topic="content_topic",
        last_result_set_items=list(items),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
        last_result_set_focus_file=items[0] if items else None,
    )


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (0, "0 B"),
        (1023, "1023 B"),
        (1024, "1.0 KiB"),
        (1536, "1.5 KiB"),
        (1024**2, "1.0 MiB"),
        (1024**3, "1.0 GiB"),
    ],
)
def test_metadata_detail_size_formatter_uses_stable_binary_units(size, expected):
    assert result_set_operations.format_file_result_set_size(size) == expected


def test_metadata_detail_preserves_order_and_maps_each_identity(tmp_path):
    notes_dir = tmp_path / "notes"
    items = ["03_说明.pdf", "01_资料.pdf", "子目录/02_记录.txt"]
    sizes = {
        "03_说明.pdf": 1024,
        "01_资料.pdf": 1024**2,
        "子目录/02_记录.txt": 1536,
    }
    times = {
        "03_说明.pdf": datetime(2026, 8, 30, 9, 0, 0),
        "01_资料.pdf": datetime(2026, 8, 29, 8, 0, 0),
        "子目录/02_记录.txt": datetime(2026, 8, 31, 10, 0, 0),
    }
    for path in items:
        _write_synthetic_file(
            notes_dir / path,
            sizes[path],
            times[path].timestamp(),
        )

    answer = result_set_operations.build_file_result_set_metadata_detail(
        " 显示详情。 ",
        items,
        entity_type="文件",
        selectable=True,
        repo_state=_repo_state(notes_dir, items, sizes=sizes, times=times),
        notes_dir=notes_dir,
    )

    assert answer is not None
    assert answer.splitlines()[0] == "当前文件结果集详情："
    assert re.search(r"#\s+类型\s+大小\s+修改时间\s+文件", answer)
    expected_rows = [
        ("1", ".pdf", "1.0 KiB", "2026-08-30 09:00:00", "03_说明.pdf"),
        ("2", ".pdf", "1.0 MiB", "2026-08-29 08:00:00", "01_资料.pdf"),
        ("3", ".txt", "1.5 KiB", "2026-08-31 10:00:00", "子目录/02_记录.txt"),
    ]
    for index, extension, size, modified_time, display_path in expected_rows:
        assert re.search(
            rf"^\s*{index}\s+{re.escape(extension)}\s+{re.escape(size)}\s+"
            rf"{re.escape(modified_time)}\s+{re.escape(display_path)}\s*$",
            answer,
            re.MULTILINE,
        )
        assert answer.count(display_path) == 1
    assert answer.index("03_说明.pdf") < answer.index("01_资料.pdf")
    assert answer.index("01_资料.pdf") < answer.index("子目录/02_记录.txt")
    assert str(notes_dir) not in answer


def test_metadata_detail_reuses_complete_repo_state_without_stat(monkeypatch, tmp_path):
    notes_dir = tmp_path / "notes"
    path = "合成资料.txt"
    repo_state = _repo_state(
        notes_dir,
        [path],
        sizes={path: 42},
        times={path: datetime(2026, 8, 31, 12, 0, 0)},
    )

    expected_path = notes_dir / path
    real_stat = Path.stat

    def guarded_stat(self, *args, **kwargs):
        if self == expected_path:
            raise AssertionError("完整 RepoState 元数据不应触发 stat")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    answer = result_set_operations.build_file_result_set_metadata_detail(
        "显示详情",
        [path],
        entity_type="文件",
        selectable=True,
        repo_state=repo_state,
        notes_dir=notes_dir,
    )

    assert answer is not None
    assert re.search(
        r"^\s*1\s+\.txt\s+42 B\s+2026-08-31 12:00:00\s+合成资料\.txt\s*$",
        answer,
        re.MULTILINE,
    )


def test_metadata_detail_never_uses_unsafe_absolute_path_as_output(tmp_path):
    notes_dir = tmp_path / "notes"
    outside_path = tmp_path / "outside" / "合成资料.pdf"

    answer = result_set_operations.build_file_result_set_metadata_detail(
        "显示详情",
        [str(outside_path)],
        entity_type="文件",
        selectable=True,
        repo_state=SimpleNamespace(paths=[]),
        notes_dir=notes_dir,
    )

    assert answer is not None
    assert re.search(
        r"^\s*1\s+\.pdf\s+未知\s+未知\s+合成资料\.pdf\s*$",
        answer,
        re.MULTILINE,
    )
    assert str(tmp_path) not in answer


def test_metadata_detail_disambiguates_duplicate_names_and_supports_windows_paths(tmp_path):
    notes_dir = tmp_path / "notes"
    items = ["项目甲/说明.pdf", r"项目乙\说明.pdf", r"目录丙\记录.TXT"]
    for index, path in enumerate(items, 1):
        normalized = path.replace("\\", "/")
        _write_synthetic_file(notes_dir / normalized, index * 10, 1_700_000_000 + index)

    answer = result_set_operations.build_file_result_set_metadata_detail(
        "显示详情",
        items,
        entity_type="文件",
        selectable=True,
        repo_state=_repo_state(notes_dir, items),
        notes_dir=notes_dir,
    )

    assert answer is not None
    assert re.search(r"^\s*1\s+\.pdf\s+10 B\s+\S+ \S+\s+项目甲/说明\.pdf\s*$", answer, re.MULTILINE)
    assert re.search(r"^\s*2\s+\.pdf\s+20 B\s+\S+ \S+\s+项目乙/说明\.pdf\s*$", answer, re.MULTILINE)
    assert re.search(r"^\s*3\s+\.txt\s+30 B\s+\S+ \S+\s+目录丙/记录\.TXT\s*$", answer, re.MULTILINE)
    assert "未知" not in answer
    assert str(notes_dir) not in answer


def test_metadata_detail_preserves_unmapped_or_deleted_item_as_unknown(tmp_path):
    notes_dir = tmp_path / "notes"
    existing = "现有资料.md"
    missing = "已移动资料.pdf"
    missing_without_type = "已删除资料"
    _write_synthetic_file(notes_dir / existing, 7, 1_700_000_000)
    repo_state = _repo_state(notes_dir, [existing])

    answer = result_set_operations.build_file_result_set_metadata_detail(
        "显示详情",
        [existing, missing, missing_without_type],
        entity_type="文件",
        selectable=True,
        repo_state=repo_state,
        notes_dir=notes_dir,
    )

    assert answer is not None
    assert re.search(r"^\s*1\s+\.md\s+7 B\s+\S+ \S+\s+现有资料\.md\s*$", answer, re.MULTILINE)
    assert re.search(
        r"^\s*2\s+\.pdf\s+未知\s+未知\s+已移动资料\.pdf\s*$",
        answer,
        re.MULTILINE,
    )
    assert re.search(
        r"^\s*3\s+未知\s+未知\s+未知\s+已删除资料\s*$",
        answer,
        re.MULTILINE,
    )
    assert str(notes_dir) not in answer


@pytest.mark.parametrize(
    ("items", "entity_type", "selectable", "question"),
    [
        (None, "文件", True, "显示详情"),
        ([], "文件", True, "显示详情"),
        (["对象甲"], "岗位", True, "显示详情"),
        (["资料甲.md"], "文件", False, "显示详情"),
        (["资料甲.md"], "文件", True, "查看详情"),
        (["资料甲.md"], "文件", True, "展开详情"),
    ],
)
def test_metadata_detail_does_not_take_over_without_exact_active_file_context(
    tmp_path,
    items,
    entity_type,
    selectable,
    question,
):
    assert result_set_operations.build_file_result_set_metadata_detail(
        question,
        items,
        entity_type=entity_type,
        selectable=selectable,
        repo_state=SimpleNamespace(paths=[]),
        notes_dir=tmp_path / "notes",
    ) is None


def _run_local_detail_turns(
    monkeypatch,
    tmp_path,
    *,
    questions: list[str],
    items: list[str],
    state: ConversationState,
):
    notes_dir = tmp_path / "notes"
    times = {}
    sizes = {}
    for index, path in enumerate(items, 1):
        size = index * 1024
        timestamp = 1_700_000_000 + (len(items) - index) * 60
        _write_synthetic_file(notes_dir / path, size, timestamp)
        sizes[path] = size
        times[path] = datetime.fromtimestamp(timestamp)
    repo_state = _repo_state(notes_dir, items, sizes=sizes, times=times)
    calls = {
        "query_router": 0,
        "query_rewrite": 0,
        "retrieval": 0,
        "local_model": 0,
        "remote_model": 0,
        "document_body_reader": 0,
    }
    inputs = iter([*questions, "q"])

    monkeypatch.setattr(chat_runtime, "_read_user_question", lambda **_kwargs: next(inputs))
    monkeypatch.setattr(chat_runtime, "_flush_pending_tty_input_unix", lambda: False)
    monkeypatch.setattr(chat_runtime, "conversation_state", state)

    def record(name):
        def forbidden(*_args, **_kwargs):
            calls[name] += 1
            raise AssertionError(f"详情本地短路前调用了 {name}")

        return forbidden

    monkeypatch.setattr(chat_runner, "analyze_question_signals", record("query_router"))
    monkeypatch.setattr(chat_runner, "detect_dialog_event", record("query_router"))
    monkeypatch.setattr(chat_runner, "resolve_route", record("query_router"))
    monkeypatch.setattr("requests.post", record("local_model"))
    monkeypatch.setattr(chat_runner, "build_search_query", record("retrieval"))
    monkeypatch.setattr(chat_runner, "build_retrieval_materials", record("retrieval"))
    monkeypatch.setattr(
        "app.retrieval_flow.query.rewrite_search_query",
        record("query_rewrite"),
    )
    monkeypatch.setattr("loaders.file_loader.read_file", record("document_body_reader"))

    chat_runtime.run_chat_loop(
        repo_state,
        SimpleNamespace(),
        _ClientSpy(calls),
        "offline-model",
        "http://127.0.0.1:9",
        "offline-model",
        _LoggerStub(),
        notes_dir=notes_dir,
        change_log_file=tmp_path / "changes.db",
        domain_dispatch_port=SimpleNamespace(),
    )
    return SimpleNamespace(calls=calls, state=chat_runtime.conversation_state)


def test_runner_details_active_set_before_all_routing_and_preserves_state(
    monkeypatch,
    tmp_path,
    capsys,
):
    items = ["03_说明.pdf", "01_资料.pdf", "02_记录.txt"]
    state = _active_state(items)
    content_anchor = (
        state.last_content_route,
        state.last_content_user_question,
        state.last_content_topic,
    )

    result = _run_local_detail_turns(
        monkeypatch,
        tmp_path,
        questions=["显示详情"],
        items=items,
        state=state,
    )

    output = capsys.readouterr().out
    assert "当前文件结果集详情：" in output
    assert result.calls == {
        "query_router": 0,
        "query_rewrite": 0,
        "retrieval": 0,
        "local_model": 0,
        "remote_model": 0,
        "document_body_reader": 0,
    }
    assert result.state.last_result_set_items == items
    assert result.state.last_result_set_entity_type == "文件"
    assert result.state.last_result_set_selectable is True
    assert result.state.last_answer_type == "enumeration_file"
    assert result.state.last_result_set_focus_file == "03_说明.pdf"
    assert result.state.last_local_topic == "list_files"
    assert (
        result.state.last_content_route,
        result.state.last_content_user_question,
        result.state.last_content_topic,
    ) == content_anchor


@pytest.mark.parametrize("local_topic", ["list_files", None])
def test_sort_then_detail_preserves_topic_order_and_third_item_selection(
    monkeypatch,
    tmp_path,
    local_topic,
):
    items = ["03_说明.pdf", "01_资料.pdf", "02_记录.txt"]
    state = _active_state(items, local_topic=local_topic)

    result = _run_local_detail_turns(
        monkeypatch,
        tmp_path,
        questions=["按文件名排列", "显示详情"],
        items=items,
        state=state,
    )

    expected_order = ["01_资料.pdf", "02_记录.txt", "03_说明.pdf"]
    assert result.state.last_result_set_items == expected_order
    assert result.state.last_local_topic == local_topic
    assert result.state.last_answer_type == "enumeration_file"
    selection = resolve_file_result_set_selection(
        "第三个文件讲了什么？",
        result.state.last_result_set_items,
    )
    assert selection is not None
    assert selection.paths == ("03_说明.pdf",)


def test_existing_full_repo_detail_modifier_keeps_time_list_semantics():
    assert classify_repo_meta_question(
        "详细看下",
        last_user_question="当前知识库有哪些文件？",
        last_local_topic="list_files",
    ) == "list_files_with_time"
