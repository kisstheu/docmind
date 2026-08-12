from __future__ import annotations

import asyncio
import copy
import gc
import inspect
import json
import os
import socket
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import app.chat_loop as runtime
import app.chat_loop_parts.runner as runner
import ask_notes
from app.dialog_state_machine import ConversationState, DialogEvent
from app.domain_dispatch_port import DomainDispatchPort, dispatch_domain_request
from app.domain_host import EmptyDomainHost, StaticDomainHost
from bootstrap.domain_composition import create_domain_host
from docmind_domain_sdk import (
    DomainPlugin,
    DomainRequest,
    DomainResult,
    FocusUpdate,
    MAX_OPTIONS_CONTAINER_ITEMS,
    MAX_OPTIONS_DEPTH,
    MAX_OPTIONS_ENCODED_BYTES,
    MAX_OPTIONS_TOTAL_KEYS,
    PluginError,
    ProtocolViolationError,
)
from docmind_recruitment_plugin import PLUGIN_ID, RecruitmentJDPlugin


_PLUGIN_ID = "org.example.neutral"
_UNSET = object()


class _Logger:
    def debug(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


class _NeutralPlugin:
    def __init__(self, result_factory):
        self._result_factory = result_factory
        self.execute_requests = []
        self.execute_loops = []
        self.other_calls = []

    async def describe(self, request):
        self.other_calls.append(("describe", request))
        raise AssertionError("describe must not be called")

    async def start(self, request):
        self.other_calls.append(("start", request))
        raise AssertionError("start must not be called")

    async def sync_sources(self, request):
        self.other_calls.append(("sync_sources", request))
        raise AssertionError("sync_sources must not be called")

    async def probe(self, request):
        self.other_calls.append(("probe", request))
        raise AssertionError("probe must not be called")

    async def execute(self, request):
        self.execute_requests.append(request)
        self.execute_loops.append(asyncio.get_running_loop())
        result = self._result_factory(request)
        if isinstance(result, BaseException):
            raise result
        return result

    async def stop(self, request):
        self.other_calls.append(("stop", request))
        raise AssertionError("stop must not be called")


def _domain_result(request, status="handled"):
    error = None
    if status in {"retryable_error", "fatal_error"}:
        error = PluginError(
            code="synthetic.failure",
            message="Synthetic plugin failure.",
            retryable=status == "retryable_error",
        )
    return DomainResult(
        request_id=request.request_id,
        plugin_id=_PLUGIN_ID,
        status=status,
        answer_markdown="Synthetic handled result." if status == "handled" else "",
        error=error,
    )


def _host_handling_only(*handled_questions):
    handled_set = set(handled_questions)
    plugin = _NeutralPlugin(
        lambda request: _domain_result(
            request,
            "handled" if request.query in handled_set else "abstain",
        )
    )
    return plugin, create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)


def _stale_interaction_state():
    return ConversationState(
        mode="content",
        last_user_question="旧问题",
        last_route="repo_meta",
        last_local_topic="list_files",
        last_answer_preview="旧文件集合回答",
        last_category_context_answer="旧分类上下文",
        last_content_user_question="旧内容问题",
        last_content_route="normal_retrieval",
        last_content_topic="旧内容主题",
        last_effective_search_query="旧检索锚点",
        last_answer_text="1. scope-a.md\n2. scope-b.md",
        last_answer_type="enumeration_file",
        last_result_set_query="旧集合问题",
        last_result_set_items=["scope-a.md", "scope-b.md"],
        last_result_set_entity_type="文件",
        last_result_set_summary_text="旧集合概括",
        last_result_set_summary_level=2,
        last_result_set_selectable=True,
        last_selected_candidate="候选项X",
        last_selected_source_files=["scope-a.md"],
        pending_action_preview="孤立的旧预览",
    )


class _SpyPort:
    def __init__(self, state: ConversationState):
        self.state = state
        self.requests = []
        self.state_snapshots = []

    def dispatch(self, request: DomainRequest):
        self.requests.append(request)
        self.state_snapshots.append(
            {
                "mode": self.state.mode,
                "last_route": self.state.last_route,
                "last_selected_source_files": list(
                    self.state.last_selected_source_files or []
                ),
                "last_result_set_items": list(self.state.last_result_set_items or []),
            }
        )
        return None


class _FakeModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text="受控模型回答")


def _state_snapshot(state: ConversationState):
    return {
        name: list(value) if isinstance(value, list) else value
        for name, value in vars(state).items()
    }


def _run_turn(
    monkeypatch,
    tmp_path,
    *,
    port,
    gate=None,
    generate=False,
    scripted_questions=None,
    initial_state=None,
    initial_focus="focus.md",
    use_real_dialog_events=False,
    use_real_contextless_guard=False,
    use_real_state_updates=False,
    material_indices=None,
    material_focus=_UNSET,
    domain_options=None,
    repo_state=None,
):
    question = "哪些文档里提到了检索策略？"
    state = initial_state or ConversationState(
        mode="idle",
        last_user_question="旧问题",
        last_route="repo_meta",
        last_local_topic="list_files",
        last_answer_preview="既有回答预览",
        last_category_context_answer="既有分类概括",
        last_content_user_question="此前内容问题",
        last_content_route="normal_retrieval",
        last_effective_search_query="旧检索查询",
        last_answer_text="既有回答文本",
        last_answer_type="enumeration_file",
        last_result_set_items=["scope-a.md", "scope-b.md"],
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
        last_selected_candidate="候选项X",
        last_selected_source_files=["scope-a.md"],
    )
    runtime.conversation_state = state
    if hasattr(port, "state"):
        port.state = state
    question_values = list(scripted_questions or [question])
    if not question_values or question_values[-1].strip().lower() not in {"q", "quit", "exit"}:
        question_values.append("q")
    questions = iter(question_values)
    captured = {
        "search": [],
        "materials": [],
        "prompts": [],
        "printed": [],
        "memory_calls": [],
        "memory_snapshots": [],
        "state_updates": [],
        "file_action_inputs": [],
        "dialog_inputs": [],
        "events": [],
        "route_inputs": [],
    }
    fake_models = _FakeModels()

    monkeypatch.setattr(runtime, "build_chat_config", lambda _repo: {"stable": True})
    monkeypatch.setattr(runtime, "_flush_pending_tty_input_unix", lambda: None)
    monkeypatch.setattr(
        runtime,
        "_read_user_question",
        lambda **_kwargs: next(questions),
    )
    def fake_handle_file_action_turn(**kwargs):
        captured["file_action_inputs"].append({
            "question": kwargs["question"],
            "state": _state_snapshot(kwargs["state"]),
            "current_focus_file": kwargs["current_focus_file"],
        })
        next_focus = kwargs["current_focus_file"]
        if len(captured["file_action_inputs"]) == 1:
            next_focus = initial_focus
        return gate == "file_action", kwargs["state"], next_focus

    monkeypatch.setattr(runner, "handle_file_action_turn", fake_handle_file_action_turn)
    real_detect_dialog_event = runner.detect_dialog_event
    real_apply_event_to_state = runner.apply_event_to_state

    def fake_detect_dialog_event(current_question, current_state, logger, **kwargs):
        captured["dialog_inputs"].append({
            "question": current_question,
            "state": _state_snapshot(current_state),
            "focused_file": kwargs.get("focused_file"),
        })
        event = (
            real_detect_dialog_event(current_question, current_state, logger, **kwargs)
            if use_real_dialog_events
            else DialogEvent(name="unknown")
        )
        captured["events"].append(event)
        return event

    monkeypatch.setattr(runner, "detect_dialog_event", fake_detect_dialog_event)
    monkeypatch.setattr(
        runner,
        "apply_event_to_state",
        real_apply_event_to_state if use_real_dialog_events else lambda current, _event: current,
    )
    if not use_real_contextless_guard:
        monkeypatch.setattr(
            runtime,
            "try_handle_contextless_followup",
            lambda **_kwargs: "守门回答" if gate == "contextless" else None,
        )
    route = gate if gate in {
        "system_capability",
        "repo_meta",
        "smalltalk",
        "out_of_scope",
    } else "normal_retrieval"
    def fake_resolve_route(current_question, event, *_args, **kwargs):
        captured["route_inputs"].append({
            "question": current_question,
            "event": event,
            "state": _state_snapshot(kwargs["state"]),
        })
        return {
            "route": route,
            "smalltalk_reply": "",
            "route_question_input": current_question,
        }

    monkeypatch.setattr(runner, "resolve_route", fake_resolve_route)
    monkeypatch.setattr(
        runtime,
        "try_handle_system_capability",
        lambda *_args: "守门回答" if gate == "system_capability" else None,
    )
    monkeypatch.setattr(
        runtime,
        "try_handle_repo_meta",
        lambda *_args, **_kwargs: (
            ("守门回答", "list_files") if gate == "repo_meta" else (None, None)
        ),
    )
    monkeypatch.setattr(
        runtime,
        "try_handle_smalltalk",
        lambda **_kwargs: "守门回答" if gate == "smalltalk" else None,
    )
    monkeypatch.setattr(
        runtime,
        "try_handle_out_of_scope",
        lambda **_kwargs: "守门回答" if gate == "out_of_scope" else None,
    )
    monkeypatch.setattr(
        runner._loop_handlers,
        "_try_answer_file_result_set_topic_summary",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runner, "determine_query_flags", lambda _question: {"skip_retrieval": False})
    monkeypatch.setattr(
        runner._loop_handlers,
        "looks_like_analytic_retrieval_question",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(
        runner,
        "resolve_repo_content_category_scope",
        lambda **_kwargs: ("稳定范围", ["scope-a.md"]),
    )

    def fake_build_search_query(**kwargs):
        captured["search"].append(kwargs)
        return "稳定检索查询", "稳定上下文锚点"

    def fake_build_materials(**kwargs):
        captured["materials"].append(kwargs)
        call_index = len(captured["materials"]) - 1
        if material_indices:
            selected_indices = material_indices[min(call_index, len(material_indices) - 1)]
        else:
            selected_indices = [2, 0]
        return {
            "current_focus_file": (
                kwargs["current_focus_file"]
                if material_focus is _UNSET
                else material_focus
            ),
            "relevant_indices": list(selected_indices),
            "inventory_candidates_text": "稳定候选",
            "context_text": "稳定上下文",
            "timeline_evidence_text": "",
        }

    monkeypatch.setattr(runner, "build_search_query", fake_build_search_query)
    monkeypatch.setattr(runner, "build_retrieval_materials", fake_build_materials)
    monkeypatch.setattr(
        runner,
        "maybe_build_related_records_answer",
        lambda **_kwargs: None if generate else "受控本地结果",
    )
    monkeypatch.setattr(runner, "maybe_build_file_location_answer", lambda **_kwargs: None)
    monkeypatch.setattr(runner, "maybe_build_direct_lookup_answer", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime,
        "try_handle_retrieval_force_local_or_empty_context",
        lambda **_kwargs: None,
    )
    def fake_build_safe_final_prompt(**kwargs):
        captured["prompts"].append(kwargs)
        return "受控提示"

    monkeypatch.setattr(runner, "build_safe_final_prompt", fake_build_safe_final_prompt)
    monkeypatch.setattr(
        runner,
        "print_answer",
        lambda answer, _start: captured["printed"].append(answer),
    )
    real_append_memory = runner.append_memory

    def capture_append_memory(memory_buffer, current_question, answer):
        captured["memory_calls"].append((current_question, answer))
        real_append_memory(memory_buffer, current_question, answer)
        captured["memory_snapshots"].append(list(memory_buffer))

    monkeypatch.setattr(runner, "append_memory", capture_append_memory)

    real_update_state_after_retrieval_answer = runner.update_state_after_retrieval_answer

    def fake_update(current, current_question, answer, current_logger, **kwargs):
        captured["state_updates"].append(answer)
        if use_real_state_updates:
            return real_update_state_after_retrieval_answer(
                current,
                current_question,
                answer,
                current_logger,
                **kwargs,
            )
        return current

    monkeypatch.setattr(runner, "update_state_after_retrieval_answer", fake_update)
    captured["runner_return"] = runner.run_chat_loop(
        SimpleNamespace() if repo_state is None else repo_state,
        None,
        SimpleNamespace(models=fake_models),
        "model-id",
        "http://local.invalid",
        "local-model",
        _Logger(),
        notes_dir=tmp_path,
        change_log_file=tmp_path / "changes.jsonl",
        domain_dispatch_port=port,
        domain_options=domain_options,
    )
    return runtime.conversation_state, captured, fake_models


def _parse_cli(monkeypatch, path: Path | None = None):
    argv = ["docmind-test"]
    if path is not None:
        argv.extend(["--domain-options-file", str(path)])
    monkeypatch.setattr(sys, "argv", argv)
    return ask_notes._parse_args()


def _assert_cli_file_failure(monkeypatch, capsys, path: Path, *secrets: str):
    with pytest.raises(SystemExit) as caught:
        _parse_cli(monkeypatch, path)

    output = capsys.readouterr()
    assert caught.value.code == 2
    assert output.out == ""
    assert output.err.startswith("usage: docmind-test")
    assert "argument --domain-options-file:" in output.err
    assert ask_notes._DOMAIN_OPTIONS_FILE_ERROR in output.err
    for secret in (str(path), path.name, *secrets):
        assert secret not in output.err
    assert "Traceback" not in output.err
    assert "ValidationError" not in output.err
    assert "repr(" not in output.err


def test_cli_without_domain_options_skips_preflight(monkeypatch):
    def fail_preflight(**_kwargs):
        raise AssertionError("preflight must not be constructed")

    monkeypatch.setattr(ask_notes, "DomainRequest", fail_preflight)

    args = _parse_cli(monkeypatch)

    assert args.domain_options is None


def test_cli_missing_domain_options_value_keeps_standard_argparse_error(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(sys, "argv", ["docmind-test", "--domain-options-file"])

    with pytest.raises(SystemExit) as caught:
        ask_notes._parse_args()

    output = capsys.readouterr()
    assert caught.value.code == 2
    assert output.out == ""
    assert "argument --domain-options-file: expected one argument" in output.err
    assert ask_notes._DOMAIN_OPTIONS_FILE_ERROR not in output.err


@pytest.mark.parametrize("with_bom", [False, True])
def test_cli_loads_strict_utf8_object_and_single_leading_bom(
    monkeypatch,
    tmp_path,
    with_bom,
):
    path = tmp_path / "synthetic-options.json"
    raw = json.dumps(
        {
            "org.example.alpha": {"enabled": True, "count": 2},
            "org.example.beta": [None, "value"],
        },
        ensure_ascii=False,
    ).encode("utf-8")
    path.write_bytes((b"\xef\xbb\xbf" if with_bom else b"") + raw)

    args = _parse_cli(monkeypatch, path)

    assert args.domain_options == {
        "org.example.alpha": {"enabled": True, "count": 2},
        "org.example.beta": [None, "value"],
    }


def test_cli_accepts_explicit_empty_object_and_exact_raw_byte_limit(
    monkeypatch,
    tmp_path,
):
    empty_path = tmp_path / "empty-object.json"
    empty_path.write_bytes(b"{}")
    assert _parse_cli(monkeypatch, empty_path).domain_options == {}

    limit_path = tmp_path / "exact-limit.json"
    limit_path.write_bytes(b"{}" + b" " * (65536 - 2))
    assert limit_path.stat().st_size == 65536
    assert _parse_cli(monkeypatch, limit_path).domain_options == {}


def test_cli_snapshot_is_isolated_read_once_and_refreshes_only_on_reload(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "lifecycle-options.json"
    first_raw = b'{"org.example.synthetic":{"revision":1}}'
    path.write_bytes(first_raw)
    before = path.stat()

    first = _parse_cli(monkeypatch, path).domain_options
    after = path.stat()
    assert path.read_bytes() == first_raw
    path.write_bytes(b'{"org.example.synthetic":{"revision":2}}')

    assert first == {"org.example.synthetic": {"revision": 1}}
    assert first["org.example.synthetic"]["revision"] == 1
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_size == before.st_size == len(first_raw)
    second = _parse_cli(monkeypatch, path).domain_options
    assert second == {"org.example.synthetic": {"revision": 2}}
    assert first == {"org.example.synthetic": {"revision": 1}}


def test_cli_snapshot_is_sdk_copy_of_parser_output(monkeypatch, tmp_path):
    path = tmp_path / "parser-copy.json"
    path.write_bytes(b"{}")
    parsed = {"org.example.synthetic": {"items": [1]}}
    monkeypatch.setattr(ask_notes.json, "loads", lambda *_args, **_kwargs: parsed)

    snapshot = _parse_cli(monkeypatch, path).domain_options
    parsed["org.example.synthetic"]["items"].append(2)

    assert snapshot == {"org.example.synthetic": {"items": [1]}}
    assert snapshot is not parsed


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b" \n\t", id="whitespace"),
        pytest.param(b'{"PRIVATE_FIELD_SENTINEL":', id="syntax"),
        pytest.param(b"[]", id="array-root"),
        pytest.param(b'"PRIVATE_VALUE_SENTINEL"', id="string-root"),
        pytest.param(b"17", id="number-root"),
        pytest.param(b"true", id="boolean-root"),
        pytest.param(b"null", id="null-root"),
        pytest.param(
            b'{"PRIVATE_FIELD_SENTINEL":1,"PRIVATE_FIELD_SENTINEL":2}',
            id="top-level-duplicate",
        ),
        pytest.param(
            b'{"outer":{"PRIVATE_FIELD_SENTINEL":1,"PRIVATE_FIELD_SENTINEL":2}}',
            id="nested-duplicate",
        ),
        pytest.param(b'{"PRIVATE_FIELD_SENTINEL":NaN}', id="nan"),
        pytest.param(b'{"PRIVATE_FIELD_SENTINEL":Infinity}', id="infinity"),
        pytest.param(b'{"PRIVATE_FIELD_SENTINEL":-Infinity}', id="negative-infinity"),
        pytest.param(b"\xff\xfe{}", id="non-utf8"),
        pytest.param(b"\xef\xbb\xbf\xef\xbb\xbf{}", id="repeated-bom"),
        pytest.param(b" \xef\xbb\xbf{}", id="non-leading-bom"),
    ],
)
def test_cli_rejects_invalid_encoding_json_and_root_without_disclosure(
    monkeypatch,
    tmp_path,
    capsys,
    raw,
):
    path = tmp_path / "PRIVATE-PATH-SENTINEL.json"
    path.write_bytes(raw)

    _assert_cli_file_failure(
        monkeypatch,
        capsys,
        path,
        "PRIVATE_FIELD_SENTINEL",
        "PRIVATE_VALUE_SENTINEL",
    )


def test_cli_rejects_file_over_raw_byte_limit_with_bounded_read(
    monkeypatch,
    tmp_path,
    capsys,
):
    path = tmp_path / "PRIVATE-LARGE-SENTINEL.json"
    path.write_bytes(b"{}" + b" " * (65537 - 2))

    _assert_cli_file_failure(monkeypatch, capsys, path, "65537")


@pytest.mark.parametrize(
    "quota",
    ["encoded_bytes", "depth", "total_keys", "container_items"],
)
def test_cli_delegates_sdk_quota_rejections_to_public_domain_request(
    monkeypatch,
    tmp_path,
    capsys,
    quota,
):
    if quota == "encoded_bytes":
        options = {"value": "x" * MAX_OPTIONS_ENCODED_BYTES}
    elif quota == "depth":
        options = _nested_options(MAX_OPTIONS_DEPTH + 1)
    elif quota == "total_keys":
        options = {f"key-{index}": index for index in range(MAX_OPTIONS_TOTAL_KEYS + 1)}
    else:
        options = {"items": [None] * MAX_OPTIONS_CONTAINER_ITEMS}
    path = tmp_path / f"PRIVATE-{quota}-SENTINEL.json"
    path.write_text(json.dumps(options), encoding="utf-8")
    assert path.stat().st_size <= 65536

    _assert_cli_file_failure(monkeypatch, capsys, path, quota, "value", "key-0")


def test_cli_delegates_recursive_json_rules_without_copying_sdk_validator(
    monkeypatch,
    tmp_path,
    capsys,
):
    path = tmp_path / "PRIVATE-CYCLE-SENTINEL.json"
    path.write_bytes(b"{}")
    cycle = {}
    cycle["cycle"] = cycle
    monkeypatch.setattr(ask_notes.json, "loads", lambda *_args, **_kwargs: cycle)

    _assert_cli_file_failure(monkeypatch, capsys, path, "cycle")


def test_cli_rejects_non_regular_files_and_fifo_swap_before_blocking_read(
    monkeypatch,
    tmp_path,
    capsys,
):
    missing = tmp_path / "PRIVATE-MISSING-SENTINEL.json"
    _assert_cli_file_failure(monkeypatch, capsys, missing, "MISSING")

    directory = tmp_path / "PRIVATE-DIRECTORY-SENTINEL"
    directory.mkdir()
    _assert_cli_file_failure(monkeypatch, capsys, directory, "DIRECTORY")

    fifo = tmp_path / "PRIVATE-FIFO-SENTINEL"
    os.mkfifo(fifo)
    _assert_cli_file_failure(monkeypatch, capsys, fifo, "FIFO")

    socket_path = tmp_path / "PRIVATE-SOCKET-SENTINEL"
    bound_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        bound_socket.bind(str(socket_path))
        _assert_cli_file_failure(monkeypatch, capsys, socket_path, "SOCKET")
    finally:
        bound_socket.close()

    device_path = tmp_path / "PRIVATE-DEVICE-SENTINEL"
    device_path.write_bytes(b"{}")
    swapped_fifo = tmp_path / "PRIVATE-SWAPPED-FIFO-SENTINEL"
    os.mkfifo(swapped_fifo)
    original_stat = os.stat

    def staged_stat(candidate, *args, **kwargs):
        if os.fspath(candidate) == os.fspath(device_path):
            return SimpleNamespace(st_mode=ask_notes.stat.S_IFCHR)
        if os.fspath(candidate) == os.fspath(swapped_fifo):
            return SimpleNamespace(st_mode=ask_notes.stat.S_IFREG)
        return original_stat(candidate, *args, **kwargs)

    monkeypatch.setattr(ask_notes.os, "stat", staged_stat)
    _assert_cli_file_failure(monkeypatch, capsys, device_path, "DEVICE")
    _assert_cli_file_failure(monkeypatch, capsys, swapped_fifo, "SWAPPED-FIFO")


@pytest.mark.parametrize("failure_stage", ["open", "fstat", "read", "close"])
def test_cli_redacts_open_read_and_close_failures(
    monkeypatch,
    tmp_path,
    capsys,
    failure_stage,
):
    path = tmp_path / f"PRIVATE-{failure_stage}-SENTINEL.json"
    path.write_bytes(b"{}")
    original_open = open

    class FailingStream:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.stream.close()
            if failure_stage == "close":
                raise OSError("PRIVATE_CLOSE_DETAIL")

        def fileno(self):
            if failure_stage == "fstat":
                raise OSError("PRIVATE_FSTAT_DETAIL")
            return self.stream.fileno()

        def read(self, size):
            assert size == 65537
            if failure_stage == "read":
                raise OSError("PRIVATE_READ_DETAIL")
            return self.stream.read(size)

    def failing_open(*args, **kwargs):
        if failure_stage == "open":
            raise PermissionError("PRIVATE_OPEN_DETAIL")
        return FailingStream(original_open(*args, **kwargs))

    monkeypatch.setattr(ask_notes, "open", failing_open, raising=False)

    _assert_cli_file_failure(
        monkeypatch,
        capsys,
        path,
        "PRIVATE_OPEN_DETAIL",
        "PRIVATE_FSTAT_DETAIL",
        "PRIVATE_READ_DETAIL",
        "PRIVATE_CLOSE_DETAIL",
    )


def test_invalid_cli_file_exits_before_runtime_initialization(
    monkeypatch,
    tmp_path,
    capsys,
):
    path = tmp_path / "PRIVATE-EARLY-EXIT-SENTINEL.json"
    path.write_bytes(b"[]")
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("runtime initialization must not run")

    monkeypatch.setattr(sys, "argv", ["docmind-test", "--domain-options-file", str(path)])
    for name in (
        "apply_environment_defaults",
        "build_logger",
        "scan_repository",
        "load_or_build_embeddings",
        "create_production_domain_host",
        "run_chat_loop",
    ):
        monkeypatch.setattr(ask_notes, name, forbidden)

    with pytest.raises(SystemExit) as caught:
        ask_notes.main()

    output = capsys.readouterr()
    assert caught.value.code == 2
    assert calls == []
    assert output.out == ""
    assert ask_notes._DOMAIN_OPTIONS_FILE_ERROR in output.err
    assert str(path) not in output.err
    assert path.name not in output.err


def test_empty_host_satisfies_sync_port_and_has_no_side_effects(capsys):
    host = EmptyDomainHost()
    request = DomainRequest(request_id="request-1", query="原始问题", source_scope=())

    assert isinstance(host, DomainDispatchPort)
    assert not inspect.iscoroutinefunction(host.dispatch)
    assert host.dispatch(request) is None
    assert request.query == "原始问题"
    assert request.source_scope == ()
    assert request.options == {}
    assert capsys.readouterr() == ("", "")
    assert isinstance(create_domain_host(), EmptyDomainHost)


def test_dispatch_preserves_query_and_transfers_options_without_semantic_changes():
    question = "  Synthetic question with exact whitespace.\n"
    options = {
        "document_rendering": {"mode": "compact", "sections": [1, 3]},
        "inventory_threshold": 0.75,
        "schedule_timezone": None,
    }
    plugin = _NeutralPlugin(_domain_result)
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    result = dispatch_domain_request(host, question, options=options)

    assert result is not None
    assert len(plugin.execute_requests) == 1
    request = plugin.execute_requests[0]
    assert request.query == question
    assert request.options == options
    assert list(request.options) == list(options)
    assert request.options is not options
    assert plugin.other_calls == []


def test_dispatch_default_options_keep_query_only_plugin_compatible():
    seen_queries = []

    def query_only_result(request):
        seen_queries.append(request.query)
        return _domain_result(request)

    plugin = _NeutralPlugin(query_only_result)
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    assert dispatch_domain_request(host, "Query-only compatibility.") is not None
    assert seen_queries == ["Query-only compatibility."]
    assert plugin.execute_requests[0].options == {}


def _nested_options(depth):
    value = "leaf"
    for _ in range(depth):
        value = {"next": value}
    return value


@pytest.mark.parametrize(
    ("options", "category"),
    [
        (
            {"value": "x" * (MAX_OPTIONS_ENCODED_BYTES - len('{"value":""}') + 1)},
            "encoded_bytes",
        ),
        (_nested_options(MAX_OPTIONS_DEPTH + 1), "depth"),
        (
            {f"key-{index}": index for index in range(MAX_OPTIONS_TOTAL_KEYS + 1)},
            "total_keys",
        ),
        (
            {"items": [None] * MAX_OPTIONS_CONTAINER_ITEMS},
            "container_items",
        ),
    ],
)
def test_static_host_revalidates_quotas_before_plugin_coroutine(
    options,
    category,
    capsys,
):
    sentinel = "PRIVATE-SENTINEL-DO-NOT-ECHO"
    plugin = _NeutralPlugin(_domain_result)
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)
    valid = DomainRequest(
        request_id="request-host-boundary",
        query="Question",
        source_scope=(),
    )
    if category == "encoded_bytes":
        options = {sentinel: next(iter(options.values()))}
    forged = valid.model_copy(update={"options": options})

    with pytest.raises(ProtocolViolationError, match=category) as caught:
        host.dispatch(forged)

    output = capsys.readouterr()
    assert plugin.execute_requests == []
    assert plugin.execute_loops == []
    assert sentinel not in str(caught.value)
    assert sentinel not in output.out
    assert sentinel not in output.err


def test_dispatch_outer_boundary_maps_host_request_violation_to_none(capsys):
    sentinel = "PRIVATE-SENTINEL-DO-NOT-ECHO"
    plugin = _NeutralPlugin(_domain_result)
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    class _MutatingForwardPort:
        def dispatch(self, request):
            request.options[sentinel] = "x" * MAX_OPTIONS_ENCODED_BYTES
            return host.dispatch(request)

    assert dispatch_domain_request(_MutatingForwardPort(), "Question") is None
    output = capsys.readouterr()
    assert plugin.execute_requests == []
    assert sentinel not in output.out
    assert sentinel not in output.err


def test_composition_factory_requires_complete_static_plugin_configuration():
    plugin = _NeutralPlugin(_domain_result)

    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    assert isinstance(plugin, DomainPlugin)
    assert isinstance(host, StaticDomainHost)
    assert isinstance(host, DomainDispatchPort)
    with pytest.raises(ValueError, match="provided together"):
        create_domain_host(plugin=plugin)
    with pytest.raises(ValueError, match="provided together"):
        create_domain_host(expected_plugin_id=_PLUGIN_ID)


def test_static_host_executes_once_with_original_request_and_returns_handled_result():
    request = DomainRequest(
        request_id="request-handled",
        query="Synthetic contract question.",
        source_scope=(),
    )
    handled_result = _domain_result(request)
    plugin = _NeutralPlugin(lambda _request: handled_result)
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    result = host.dispatch(request)

    assert result is handled_result
    assert result.status == "handled"
    assert plugin.execute_requests == [request]
    assert plugin.execute_requests[0] is request
    assert plugin.other_calls == []
    assert len(plugin.execute_loops) == 1
    assert plugin.execute_loops[0].is_closed()


@pytest.mark.parametrize("status", ["abstain", "retryable_error", "fatal_error"])
def test_static_host_maps_each_non_handled_status_to_none(status):
    plugin = _NeutralPlugin(lambda request: _domain_result(request, status))
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)
    request = DomainRequest(
        request_id=f"request-{status}",
        query="Synthetic status question.",
        source_scope=(),
    )

    assert host.dispatch(request) is None
    assert plugin.execute_requests == [request]
    assert plugin.other_calls == []
    assert plugin.execute_loops[0].is_closed()


def test_static_host_uses_a_distinct_closed_event_loop_for_each_dispatch():
    plugin = _NeutralPlugin(_domain_result)
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    for sequence in (1, 2):
        request = DomainRequest(
            request_id=f"request-loop-{sequence}",
            query="Synthetic loop question.",
            source_scope=(),
        )
        assert host.dispatch(request).request_id == request.request_id

    assert len(plugin.execute_requests) == 2
    assert len(plugin.execute_loops) == 2
    assert plugin.execute_loops[0] is not plugin.execute_loops[1]
    assert all(loop.is_closed() for loop in plugin.execute_loops)
    assert plugin.other_calls == []


def test_static_host_closes_event_loop_after_execute_exception_and_outer_boundary_falls_back():
    plugin = _NeutralPlugin(lambda _request: RuntimeError("synthetic execute failure"))
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    result = dispatch_domain_request(host, "Synthetic exception question.")

    assert result is None
    assert len(plugin.execute_requests) == 1
    assert len(plugin.execute_loops) == 1
    assert plugin.execute_loops[0].is_closed()
    assert asyncio.all_tasks(plugin.execute_loops[0]) == set()
    assert plugin.other_calls == []


def test_static_host_refuses_running_loop_before_creating_execute_coroutine():
    plugin = _NeutralPlugin(_domain_result)
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    async def dispatch_inside_running_loop():
        return dispatch_domain_request(host, "Synthetic nested-loop question.")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = asyncio.run(dispatch_inside_running_loop())
        gc.collect()

    assert result is None
    assert plugin.execute_requests == []
    assert plugin.execute_loops == []
    assert plugin.other_calls == []
    assert not any("was never awaited" in str(item.message) for item in caught)


@pytest.mark.parametrize("mismatch", ["request_id", "plugin_id"])
def test_static_host_protocol_boundary_mismatch_falls_back_instead_of_succeeding(mismatch):
    def invalid_result(request):
        result = _domain_result(request)
        if mismatch == "request_id":
            return result.model_copy(update={"request_id": "request-other"})
        return result.model_copy(update={"plugin_id": "org.example.other"})

    plugin = _NeutralPlugin(invalid_result)
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    assert dispatch_domain_request(host, "Synthetic boundary question.") is None
    assert len(plugin.execute_requests) == 1
    assert plugin.execute_loops[0].is_closed()
    assert plugin.other_calls == []


def test_composition_root_creates_and_injects_host(monkeypatch, tmp_path):
    captured = {}
    host = EmptyDomainHost()
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_sentence = ModuleType("sentence_transformers")
    fake_sentence.SentenceTransformer = lambda *_args, **_kwargs: object()
    fake_google = ModuleType("google")
    fake_google.__path__ = []
    fake_genai = ModuleType("google.genai")
    fake_genai.Client = lambda **_kwargs: object()
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(
        ask_notes,
        "_parse_args",
        lambda: SimpleNamespace(notes_dir=None, domain_options=None),
    )
    monkeypatch.setattr(ask_notes, "apply_environment_defaults", lambda: None)
    monkeypatch.setattr(ask_notes, "build_logger", _Logger)
    monkeypatch.setattr(ask_notes, "_resolve_notes_dir", lambda *_args: tmp_path)
    monkeypatch.setattr(ask_notes, "_resolve_cache_file", lambda *_args: tmp_path / "cache.npz")
    monkeypatch.setattr(ask_notes, "_resolve_change_log_file", lambda *_args: tmp_path / "changes.db")
    monkeypatch.setattr(ask_notes, "build_debug_question_recorder", lambda **_kwargs: object())
    monkeypatch.setattr(ask_notes, "scan_repository", lambda *_args: object())
    monkeypatch.setattr(ask_notes, "load_or_build_embeddings", lambda *_args: object())
    monkeypatch.setattr(ask_notes, "create_production_domain_host", lambda: host)
    monkeypatch.setattr(
        ask_notes,
        "run_chat_loop",
        lambda *args, **kwargs: captured.update({"args": args, "kwargs": kwargs}),
    )

    ask_notes.main()

    assert captured["kwargs"]["domain_dispatch_port"] is host
    assert captured["kwargs"]["domain_options"] is None
    assert isinstance(captured["kwargs"]["domain_dispatch_port"], DomainDispatchPort)


def test_main_wires_real_cli_loader_snapshot_to_runner(monkeypatch, tmp_path):
    captured = {}
    path = tmp_path / "synthetic-main-options.json"
    path.write_text(
        json.dumps(
            {
                "org.example.alpha": {"enabled": True},
                "org.example.beta": {"threshold": 3},
            }
        ),
        encoding="utf-8",
    )
    host = EmptyDomainHost()
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    fake_torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    fake_sentence = ModuleType("sentence_transformers")
    fake_sentence.SentenceTransformer = lambda *_args, **_kwargs: object()
    fake_google = ModuleType("google")
    fake_google.__path__ = []
    fake_genai = ModuleType("google.genai")
    fake_genai.Client = lambda **_kwargs: object()
    fake_google.genai = fake_genai
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_sentence)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setattr(
        sys,
        "argv",
        ["docmind-test", "--domain-options-file", str(path)],
    )
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr(ask_notes, "apply_environment_defaults", lambda: None)
    monkeypatch.setattr(ask_notes, "build_logger", _Logger)
    monkeypatch.setattr(ask_notes, "_resolve_notes_dir", lambda *_args: tmp_path)
    monkeypatch.setattr(ask_notes, "_resolve_cache_file", lambda *_args: tmp_path / "cache.npz")
    monkeypatch.setattr(ask_notes, "_resolve_change_log_file", lambda *_args: tmp_path / "changes.db")
    monkeypatch.setattr(ask_notes, "build_debug_question_recorder", lambda **_kwargs: object())
    monkeypatch.setattr(ask_notes, "scan_repository", lambda *_args: object())
    monkeypatch.setattr(ask_notes, "load_or_build_embeddings", lambda *_args: object())
    monkeypatch.setattr(ask_notes, "create_production_domain_host", lambda: host)
    monkeypatch.setattr(
        ask_notes,
        "run_chat_loop",
        lambda *args, **kwargs: captured.update({"args": args, "kwargs": kwargs}),
    )

    ask_notes.main()

    assert captured["kwargs"]["domain_options"] == {
        "org.example.alpha": {"enabled": True},
        "org.example.beta": {"threshold": 3},
    }
    assert captured["kwargs"]["domain_dispatch_port"] is host


def test_empty_host_dispatch_preserves_normal_retrieval_chain(monkeypatch, tmp_path):
    state = ConversationState()
    port = _SpyPort(state)
    state, captured, fake_models = _run_turn(monkeypatch, tmp_path, port=port)

    assert len(port.requests) == 1
    request = port.requests[0]
    assert request.query == "哪些文档里提到了检索策略？"
    assert request.source_scope == ()
    assert port.state_snapshots == [{
        "mode": "idle",
        "last_route": "repo_meta",
        "last_selected_source_files": ["scope-a.md"],
        "last_result_set_items": ["scope-a.md", "scope-b.md"],
    }]
    assert len(captured["search"]) == len(captured["materials"]) == 1
    assert captured["search"][0]["question"] == request.query
    assert captured["search"][0]["last_selected_source_files"] == ["scope-a.md"]
    assert captured["materials"][0]["question"] == request.query
    assert captured["materials"][0]["allowed_paths"] == ["scope-a.md"]
    assert captured["materials"][0]["selected_source_files"] == ["scope-a.md"]
    assert captured["materials"][0]["current_focus_file"] == "focus.md"
    assert captured["materials"][0]["last_relevant_indices"] == []
    assert captured["printed"] == captured["state_updates"] == ["受控本地结果"]
    assert fake_models.calls == []
    assert state.last_route == "normal_retrieval"


@pytest.mark.parametrize(
    "file_options",
    [
        None,
        {},
        {
            "org.example.alpha": {"items": [1, 2]},
            "org.example.beta": {"enabled": True},
        },
    ],
)
def test_runner_preserves_none_call_shape_and_explicit_options(
    monkeypatch,
    tmp_path,
    file_options,
):
    if file_options is None:
        snapshot = None
    else:
        path = tmp_path / "runner-call-shape.json"
        path.write_text(json.dumps(file_options), encoding="utf-8")
        snapshot = _parse_cli(monkeypatch, path).domain_options
    before = copy.deepcopy(snapshot)
    calls = []

    def capture_dispatch(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(runner, "dispatch_domain_request", capture_dispatch)
    _run_turn(
        monkeypatch,
        tmp_path,
        port=EmptyDomainHost(),
        domain_options=snapshot,
    )

    assert len(calls) == 1
    assert len(calls[0][0]) == 2
    if snapshot is None:
        assert calls[0][1] == {}
    else:
        assert calls[0][1] == {"options": snapshot}
        assert calls[0][1]["options"] is snapshot
    assert snapshot == before


def test_runner_real_dispatch_creates_independent_sdk_copy_per_request(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "runner-copy-options.json"
    path.write_text(
        json.dumps({"org.example.synthetic": {"items": [1, 2]}}),
        encoding="utf-8",
    )
    snapshot = _parse_cli(monkeypatch, path).domain_options
    before = copy.deepcopy(snapshot)
    plugin = _NeutralPlugin(lambda request: _domain_result(request, "abstain"))
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=["合成领域请求甲", "合成领域请求乙"],
        domain_options=snapshot,
    )

    assert len(plugin.execute_requests) == 2
    first, second = plugin.execute_requests
    assert first.options == second.options == snapshot == before
    assert first.options is not snapshot
    assert second.options is not snapshot
    assert first.options is not second.options
    assert first.options["org.example.synthetic"] is not snapshot["org.example.synthetic"]
    assert first.options["org.example.synthetic"] is not second.options["org.example.synthetic"]
    assert first.options["org.example.synthetic"]["items"] is not second.options[
        "org.example.synthetic"
    ]["items"]
    assert snapshot == before


def test_runner_preserves_normalized_query_and_keeps_options_out_of_query(
    monkeypatch,
    tmp_path,
):
    raw_question = "请 看 下 合成记录？"
    path = tmp_path / "runner-query-options.json"
    path.write_text(
        json.dumps({"org.example.synthetic": {"marker": "OPTIONS_QUERY_SENTINEL"}}),
        encoding="utf-8",
    )
    snapshot = _parse_cli(monkeypatch, path).domain_options
    state = ConversationState()
    port = _SpyPort(state)

    _run_turn(
        monkeypatch,
        tmp_path,
        port=port,
        scripted_questions=[raw_question],
        domain_options=snapshot,
    )

    assert len(port.requests) == 1
    expected = runner.normalize_colloquial_question(raw_question)
    assert port.requests[0].query == expected
    assert port.requests[0].options == snapshot
    assert "OPTIONS_QUERY_SENTINEL" not in port.requests[0].query
    assert "org.example.synthetic" not in port.requests[0].query


def test_runner_preserves_corrected_result_set_query_with_options(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "runner-correction-options.json"
    path.write_text(
        json.dumps({"org.example.synthetic": {"enabled": True}}),
        encoding="utf-8",
    )
    snapshot = _parse_cli(monkeypatch, path).domain_options
    state = ConversationState(
        mode="content",
        last_user_question="查看第4个文件",
        last_answer_text="当前结果集中只有 2 个文件，请选择第 1～2 个",
        last_answer_type="enumeration_file",
        last_result_set_items=["scope-a.md", "scope-b.md"],
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
    )
    port = _SpyPort(state)

    _run_turn(
        monkeypatch,
        tmp_path,
        port=port,
        scripted_questions=["改成第2个文件"],
        initial_state=state,
        domain_options=snapshot,
    )

    assert len(port.requests) == 1
    assert port.requests[0].query == "查看第2个文件"
    assert port.requests[0].options == snapshot


@pytest.mark.parametrize(
    "gate",
    ["file_action", "contextless", "system_capability", "repo_meta", "smalltalk", "out_of_scope"],
)
def test_common_guards_do_not_dispatch(monkeypatch, tmp_path, gate):
    state = ConversationState()
    port = _SpyPort(state)

    _run_turn(
        monkeypatch,
        tmp_path,
        port=port,
        gate=gate,
        domain_options={"org.example.synthetic": {"enabled": True}},
    )

    assert port.requests == []


def test_empty_host_reaches_controlled_generation_boundary(monkeypatch, tmp_path):
    state = ConversationState()
    port = _SpyPort(state)
    options = {
        "org.example.synthetic": {"marker": "MODEL-PROMPT-OPTIONS-SENTINEL"}
    }

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=port,
        generate=True,
        domain_options=options,
    )

    assert len(port.requests) == 1
    assert len(fake_models.calls) == 1
    assert fake_models.calls[0] == {
        "model": "model-id",
        "contents": "受控提示",
        "config": {"stable": True},
    }
    assert captured["materials"][0]["question"] == port.requests[0].query
    assert captured["materials"][0]["selected_source_files"] == ["scope-a.md"]
    assert captured["printed"] == captured["state_updates"] == ["受控模型回答"]
    assert "MODEL-PROMPT-OPTIONS-SENTINEL" not in repr(captured["prompts"])
    assert "org.example.synthetic" not in repr(captured["prompts"])
    assert "MODEL-PROMPT-OPTIONS-SENTINEL" not in repr(vars(state))
    for generated_path in tmp_path.iterdir():
        if generated_path.is_file():
            assert b"MODEL-PROMPT-OPTIONS-SENTINEL" not in generated_path.read_bytes()
    assert state.last_route == "normal_retrieval"


def test_static_minimal_handled_result_is_presented_once_and_short_circuits_retrieval(
    monkeypatch,
    tmp_path,
    capsys,
):
    plugin = _NeutralPlugin(_domain_result)
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    options = {"org.example.synthetic": {"enabled": True}}
    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        domain_options=options,
    )
    output = capsys.readouterr()

    assert len(plugin.execute_requests) == 1
    request = plugin.execute_requests[0]
    assert isinstance(request, DomainRequest)
    assert request.query == "哪些文档里提到了检索策略？"
    assert request.source_scope == ()
    assert request.options == options
    assert plugin.other_calls == []
    assert plugin.execute_loops[0].is_closed()
    assert captured["printed"] == ["Synthetic handled result."]
    assert captured["memory_calls"] == [
        ("哪些文档里提到了检索策略？", "Synthetic handled result.")
    ]
    assert captured["memory_snapshots"] == [[
        "用户问：哪些文档里提到了检索策略？",
        "AI答：Synthetic handled result.",
    ]]
    assert captured["runner_return"] is None
    assert state == ConversationState()
    assert captured["search"] == []
    assert captured["materials"] == []
    assert captured["prompts"] == []
    assert captured["state_updates"] == []
    assert output.err == ""
    assert "[远程模型生成]" not in output.out
    assert "插件状态" not in output.out
    assert fake_models.calls == []


@pytest.mark.parametrize("status", ["abstain", "retryable_error", "fatal_error"])
def test_runner_unhandled_statuses_preserve_normal_retrieval_behavior(
    monkeypatch,
    tmp_path,
    status,
):
    plugin = _NeutralPlugin(lambda request: _domain_result(request, status))
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        domain_options={"org.example.synthetic": {"enabled": True}},
    )

    assert len(plugin.execute_requests) == 1
    assert captured["printed"] == captured["state_updates"] == ["受控本地结果"]
    assert len(captured["search"]) == len(captured["materials"]) == 1
    assert captured["prompts"] == []
    assert fake_models.calls == []
    assert state.last_route == "normal_retrieval"
    assert state.last_result_set_items == ["scope-a.md", "scope-b.md"]
    assert state.last_selected_candidate == "候选项X"


def test_runner_plugin_exception_preserves_existing_fallback(monkeypatch, tmp_path):
    plugin = _NeutralPlugin(lambda _request: RuntimeError("synthetic execute failure"))
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        domain_options={"org.example.synthetic": {"enabled": True}},
    )

    assert len(plugin.execute_requests) == 1
    assert captured["printed"] == captured["state_updates"] == ["受控本地结果"]
    assert len(captured["search"]) == len(captured["materials"]) == 1
    assert captured["prompts"] == []
    assert fake_models.calls == []
    assert state.last_route == "normal_retrieval"


@pytest.mark.parametrize("variant", ["warnings", "focus_clear"])
def test_adjacent_legal_handled_profiles_fall_back_without_reset(
    monkeypatch,
    tmp_path,
    variant,
):
    def adjacent_result(request):
        result = _domain_result(request)
        if variant == "warnings":
            return result.model_copy(update={"warnings": ("Synthetic warning.",)})
        return result.model_copy(update={"focus_update": FocusUpdate(mode="clear")})

    plugin = _NeutralPlugin(adjacent_result)
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    state, captured, fake_models = _run_turn(monkeypatch, tmp_path, port=host)

    assert captured["printed"] == ["受控本地结果"]
    assert "Synthetic handled result." not in captured["printed"]
    assert captured["memory_calls"] == [
        ("哪些文档里提到了检索策略？", "受控本地结果")
    ]
    assert len(captured["search"]) == len(captured["materials"]) == 1
    assert state.last_route == "normal_retrieval"
    assert state.last_result_set_items == ["scope-a.md", "scope-b.md"]
    assert state.last_selected_candidate == "候选项X"
    assert fake_models.calls == []


@pytest.mark.parametrize("malformed_kind", ["request_id_mismatch", "non_domain_result"])
def test_runner_malformed_or_non_domain_result_uses_existing_fallback(
    monkeypatch,
    tmp_path,
    malformed_kind,
):
    if malformed_kind == "request_id_mismatch":
        plugin = _NeutralPlugin(
            lambda request: _domain_result(request).model_copy(
                update={"request_id": "request-other"}
            )
        )
        port = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)
    else:
        port = SimpleNamespace(dispatch=lambda _request: {"status": "handled"})

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=port,
        domain_options={"org.example.synthetic": {"enabled": True}},
    )

    assert captured["printed"] == ["受控本地结果"]
    assert captured["memory_calls"] == [
        ("哪些文档里提到了检索策略？", "受控本地结果")
    ]
    assert len(captured["search"]) == len(captured["materials"]) == 1
    assert state.last_route == "normal_retrieval"
    assert state.last_result_set_items == ["scope-a.md", "scope-b.md"]
    assert fake_models.calls == []


def test_handled_reset_prevents_old_result_set_from_reaching_next_turn(
    monkeypatch,
    tmp_path,
):
    plugin, host = _host_handling_only("领域请求甲")

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=["领域请求甲", "这两个分别对应哪些名称？"],
        initial_state=_stale_interaction_state(),
        initial_focus="scope-a.md",
        use_real_dialog_events=True,
        generate=True,
    )

    assert len(plugin.execute_requests) == 2
    assert captured["events"][1].name not in {
        "result_set_followup",
        "result_set_expansion_followup",
        "synthesis_request",
    }
    assert captured["dialog_inputs"][1]["state"] == _state_snapshot(ConversationState())
    assert captured["dialog_inputs"][1]["focused_file"] is None
    assert len(captured["search"]) == len(captured["materials"]) == 1
    assert captured["search"][0]["last_result_set_items"] is None
    assert captured["search"][0]["last_result_set_entity_type"] is None
    assert captured["materials"][0]["allowed_paths"] is None
    assert captured["prompts"][0]["result_set_items"] is None
    assert captured["printed"] == ["Synthetic handled result.", "受控模型回答"]
    assert len(fake_models.calls) == 1
    assert state.last_result_set_items is None


def test_handled_reset_prevents_old_selected_candidate_followup(
    monkeypatch,
    tmp_path,
):
    plugin, host = _host_handling_only("领域请求甲")

    _, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=["领域请求甲", "详细分析一下"],
        initial_state=_stale_interaction_state(),
        initial_focus="scope-a.md",
        use_real_dialog_events=True,
        generate=True,
    )

    assert len(plugin.execute_requests) == 2
    assert captured["events"][1].name != "selected_candidate_followup"
    assert captured["dialog_inputs"][1]["state"]["last_selected_candidate"] is None
    assert captured["dialog_inputs"][1]["state"]["last_selected_source_files"] is None
    assert captured["dialog_inputs"][1]["focused_file"] is None
    assert captured["search"][0]["last_selected_candidate"] is None
    assert captured["search"][0]["last_selected_source_files"] is None
    assert captured["materials"][0]["selected_source_files"] is None
    assert captured["prompts"][0]["selected_candidate"] is None
    assert captured["prompts"][0]["selected_source_files"] is None
    assert len(fake_models.calls) == 1


def test_handled_reset_clears_old_file_focus_and_relevant_indices(
    monkeypatch,
    tmp_path,
):
    plugin, host = _host_handling_only("领域请求甲")

    _, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=["预热普通检索", "领域请求甲", "新的明确检索请求"],
        initial_state=_stale_interaction_state(),
        initial_focus="旧文件.md",
        use_real_dialog_events=True,
        generate=True,
        material_indices=[[8, 3], [5]],
    )

    assert len(plugin.execute_requests) == 3
    assert captured["file_action_inputs"][1]["current_focus_file"] == "旧文件.md"
    assert captured["file_action_inputs"][2]["current_focus_file"] is None
    assert captured["dialog_inputs"][2]["focused_file"] is None
    assert len(captured["search"]) == len(captured["materials"]) == 2
    assert captured["search"][1]["last_relevant_indices"] == []
    assert captured["materials"][1]["current_focus_file"] is None
    assert captured["materials"][1]["last_relevant_indices"] == []
    assert len(fake_models.calls) == 2


def test_handled_reset_prevents_old_route_and_content_context_inheritance(
    monkeypatch,
    tmp_path,
):
    plugin, host = _host_handling_only("领域请求甲")

    _, captured, _ = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=["领域请求甲", "继续"],
        initial_state=_stale_interaction_state(),
        initial_focus="scope-a.md",
        use_real_dialog_events=True,
    )

    second_dialog_state = captured["dialog_inputs"][1]["state"]
    second_route_state = captured["route_inputs"][1]["state"]
    assert captured["events"][1].merged_query is None
    for field in (
        "last_route",
        "last_content_route",
        "last_content_user_question",
        "last_content_topic",
        "last_effective_search_query",
        "last_local_topic",
        "last_category_context_answer",
    ):
        assert second_dialog_state[field] is None
        assert second_route_state[field] is None
    assert captured["search"][0]["last_effective_search_query"] is None
    assert captured["search"][0]["last_user_question"] is None


def test_handled_memory_preserves_prior_history_and_does_not_write_answer_state(
    monkeypatch,
    tmp_path,
):
    plugin, host = _host_handling_only("领域请求甲")

    state, captured, _ = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=["普通问题", "领域请求甲"],
    )

    assert len(plugin.execute_requests) == 2
    assert captured["memory_snapshots"][-1] == [
        "用户问：普通问题",
        "AI答：受控本地结果",
        "用户问：领域请求甲",
        "AI答：Synthetic handled result.",
    ]
    assert state == ConversationState()
    assert state.last_answer_text is None
    assert state.last_answer_preview is None
    assert state.last_answer_type is None


def test_next_turn_can_dispatch_again_and_consume_another_minimal_result(
    monkeypatch,
    tmp_path,
):
    plugin, host = _host_handling_only("领域请求甲", "领域请求乙")

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=["领域请求甲", "领域请求乙"],
        initial_state=_stale_interaction_state(),
        initial_focus="scope-a.md",
    )

    assert [request.query for request in plugin.execute_requests] == ["领域请求甲", "领域请求乙"]
    assert captured["printed"] == [
        "Synthetic handled result.",
        "Synthetic handled result.",
    ]
    assert captured["memory_calls"] == [
        ("领域请求甲", "Synthetic handled result."),
        ("领域请求乙", "Synthetic handled result."),
    ]
    assert captured["search"] == captured["materials"] == captured["prompts"] == []
    assert fake_models.calls == []
    assert captured["file_action_inputs"][1]["current_focus_file"] is None
    assert state == ConversationState()


@pytest.mark.parametrize("pending_action_type", ["rename", "delete", "organize"])
def test_valid_pending_action_still_precedes_domain_dispatch(
    monkeypatch,
    tmp_path,
    pending_action_type,
):
    pending_state = ConversationState(
        pending_action_type=pending_action_type,
        pending_action_source_path="旧文件.md",
        pending_action_target_path="新文件.md",
        pending_action_requested_text="原动作请求",
        pending_action_preview="待确认预览",
        pending_action_payload=(
            '{"root_rel_path": "已整理", "moves": '
            '[{"source_rel_path": "旧文件.md", "target_rel_path": "新文件.md"}]}'
        ),
    )
    plugin = _NeutralPlugin(_domain_result)
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        gate="file_action",
        scripted_questions=["确认"],
        initial_state=pending_state,
        initial_focus="旧文件.md",
    )

    assert captured["file_action_inputs"][0]["state"]["pending_action_type"] == pending_action_type
    assert plugin.execute_requests == []
    assert captured["events"] == []
    assert captured["search"] == captured["materials"] == captured["prompts"] == []
    assert fake_models.calls == []
    assert state.pending_action_type == pending_action_type


_RESOLVED_JD_PATH = "招聘/某公司A-Python岗位.md"
_LONG_SYNTHETIC_JD = "\n".join(
    [
        "职位名称：Python 应用开发工程师",
        "岗位职责：负责某公司A的合成知识库功能与可重复验证记录。",
        "任职要求：",
        "1. 熟悉 Python；",
        "2. 了解 FastAPI；",
        "3. 具有合成检索项目经验。",
        "工作地点：示例城市A",
        "工作制：双休",
        *[
            f"补充记录{index:03d}：此段仅为脱敏合成占位信息。"
            for index in range(240)
        ],
        "是否外包：是",
    ]
)
_RECRUITMENT_OPTIONS = {
    PLUGIN_ID: {
        "schema_version": "1.0",
        "explicit_rules": {
            "allow_outsourcing": False,
            "require_double_weekends": True,
        },
    },
    "org.example.synthetic": {"marker": "coexisting-namespace"},
}


def _repo_with_documents(paths, docs):
    return SimpleNamespace(paths=list(paths), docs=list(docs))


def _capture_recruitment_execute(monkeypatch):
    observed = []
    original_execute = RecruitmentJDPlugin.execute

    async def capture_execute(self, request):
        observed.append(request)
        return await original_execute(self, request)

    monkeypatch.setattr(RecruitmentJDPlugin, "execute", capture_execute)
    return observed


def _selectable_jd_state(paths, *, focus_file=None):
    answer = "\n".join(
        f"{index}. {path}" for index, path in enumerate(paths, 1)
    )
    return ConversationState(
        mode="content",
        last_user_question="列出合成资料",
        last_route="normal_retrieval",
        last_content_user_question="列出合成资料",
        last_content_route="normal_retrieval",
        last_effective_search_query="合成资料",
        last_answer_text=answer,
        last_answer_preview=answer,
        last_answer_type="enumeration_file",
        last_result_set_items=list(paths),
        last_result_set_entity_type="文件",
        last_result_set_selectable=True,
        last_result_set_focus_file=focus_file,
    )


def test_first_turn_unique_file_reference_dispatches_complete_repo_document(
    monkeypatch,
    tmp_path,
):
    observed = _capture_recruitment_execute(monkeypatch)
    host = create_domain_host(
        plugin=RecruitmentJDPlugin(),
        expected_plugin_id=PLUGIN_ID,
    )
    repo_state = _repo_with_documents(
        [_RESOLVED_JD_PATH],
        [_LONG_SYNTHETIC_JD],
    )
    question = "看看某公司A-Python岗位怎么样"

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=[question],
        initial_state=ConversationState(),
        initial_focus=None,
        use_real_dialog_events=True,
        use_real_state_updates=True,
        material_focus=_RESOLVED_JD_PATH,
        material_indices=[[0]],
        domain_options=_RECRUITMENT_OPTIONS,
        repo_state=repo_state,
        generate=True,
    )

    assert len(_LONG_SYNTHETIC_JD) > 3_000
    assert [request.query for request in observed] == [
        question,
        repo_state.docs[0],
    ]
    assert observed[1].query == _LONG_SYNTHETIC_JD
    assert observed[1].options == _RECRUITMENT_OPTIONS
    assert captured["printed"][0].startswith("## 单 JD 显式规则比较\n\n")
    assert "JD 明确为外包" in captured["printed"][0]
    assert captured["state_updates"] == captured["printed"]
    assert captured["prompts"] == []
    assert fake_models.calls == []
    assert state.last_route == "normal_retrieval"
    assert state.last_content_user_question == question
    assert state.last_answer_text == captured["printed"][0]


def test_result_set_ordinal_dispatches_selected_full_document(
    monkeypatch,
    tmp_path,
):
    observed = _capture_recruitment_execute(monkeypatch)
    host = create_domain_host(
        plugin=RecruitmentJDPlugin(),
        expected_plugin_id=PLUGIN_ID,
    )
    paths = ["招聘/资料甲.md", _RESOLVED_JD_PATH, "招聘/资料丙.md"]
    repo_state = _repo_with_documents(
        paths,
        ["合成资料甲。", _LONG_SYNTHETIC_JD, "合成资料丙。"],
    )
    question = "第 2 个怎么样"

    _state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=[question],
        initial_state=_selectable_jd_state(paths),
        initial_focus=None,
        use_real_dialog_events=True,
        material_indices=[[0]],
        domain_options=_RECRUITMENT_OPTIONS,
        repo_state=repo_state,
        generate=True,
    )

    assert [request.query for request in observed] == [question, repo_state.docs[1]]
    assert observed[1].query != question
    assert observed[1].options == _RECRUITMENT_OPTIONS
    assert captured["materials"][0]["allowed_paths"] == {_RESOLVED_JD_PATH}
    assert captured["printed"][0].startswith("## 单 JD 显式规则比较\n\n")
    assert captured["prompts"] == []
    assert fake_models.calls == []


def test_focus_continuation_dispatches_focused_full_document(
    monkeypatch,
    tmp_path,
):
    observed = _capture_recruitment_execute(monkeypatch)
    host = create_domain_host(
        plugin=RecruitmentJDPlugin(),
        expected_plugin_id=PLUGIN_ID,
    )
    repo_state = _repo_with_documents(
        [_RESOLVED_JD_PATH],
        [_LONG_SYNTHETIC_JD],
    )
    state = _selectable_jd_state(
        [_RESOLVED_JD_PATH],
        focus_file=_RESOLVED_JD_PATH,
    )
    question = "这个 JD 符合我的求职条件吗"

    _state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=[question],
        initial_state=state,
        initial_focus=_RESOLVED_JD_PATH,
        use_real_dialog_events=True,
        material_indices=[[0]],
        domain_options=_RECRUITMENT_OPTIONS,
        repo_state=repo_state,
        generate=True,
    )

    assert [request.query for request in observed] == [question, repo_state.docs[0]]
    assert observed[1].options == _RECRUITMENT_OPTIONS
    assert captured["materials"][0]["allowed_paths"] == {_RESOLVED_JD_PATH}
    assert captured["printed"][0].startswith("## 单 JD 显式规则比较\n\n")
    assert captured["prompts"] == []
    assert fake_models.calls == []


def test_resolved_jd_handled_preserves_focus_for_demonstrative_continuation(
    monkeypatch,
    tmp_path,
):
    observed = _capture_recruitment_execute(monkeypatch)
    host = create_domain_host(
        plugin=RecruitmentJDPlugin(),
        expected_plugin_id=PLUGIN_ID,
    )
    repo_state = _repo_with_documents(
        [_RESOLVED_JD_PATH],
        [_LONG_SYNTHETIC_JD],
    )
    questions = [
        "第 1 个怎么样",
        "这个 JD 符合我的求职条件吗",
    ]

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=questions,
        initial_state=_selectable_jd_state([_RESOLVED_JD_PATH]),
        initial_focus=None,
        use_real_dialog_events=True,
        use_real_contextless_guard=True,
        use_real_state_updates=True,
        material_indices=[[0], [0]],
        domain_options={},
        repo_state=repo_state,
        generate=True,
    )

    assert [request.query for request in observed] == [
        questions[0],
        repo_state.docs[0],
        questions[1],
        repo_state.docs[0],
    ]
    assert captured["dialog_inputs"][1]["focused_file"] == _RESOLVED_JD_PATH
    assert captured["events"][1].name == "content_followup"
    assert captured["materials"][1]["allowed_paths"] == {_RESOLVED_JD_PATH}
    assert [answer.startswith("## JD 明确约束\n\n") for answer in captured["printed"]] == [
        True,
        True,
    ]
    assert state.last_result_set_focus_file == _RESOLVED_JD_PATH
    assert captured["prompts"] == []
    assert fake_models.calls == []


def test_generated_ordinal_dispatches_proven_backing_and_preserves_domain_focus(
    monkeypatch,
    tmp_path,
):
    observed = _capture_recruitment_execute(monkeypatch)
    host = create_domain_host(
        plugin=RecruitmentJDPlugin(),
        expected_plugin_id=PLUGIN_ID,
    )
    paths = ["招聘/资料甲.md", _RESOLVED_JD_PATH, "招聘/资料丙.md"]
    repo_state = _repo_with_documents(
        paths,
        ["合成资料甲。", _LONG_SYNTHETIC_JD, "合成资料丙。"],
    )
    generated_answer = "1. 岗位甲\n2. 岗位乙"
    state = ConversationState(
        mode="content",
        last_user_question="有哪些岗位？",
        last_route="normal_retrieval",
        last_content_user_question="有哪些岗位？",
        last_content_route="normal_retrieval",
        last_effective_search_query="合成岗位",
        last_answer_text=generated_answer,
        last_answer_preview=generated_answer,
        last_answer_type=None,
        last_result_set_items=list(paths),
        last_result_set_entity_type="文件",
        last_result_set_selectable=False,
        last_generated_result_items=["岗位甲", "岗位乙"],
        last_generated_result_source_candidates=list(paths),
        last_generated_result_source_hits=[[_RESOLVED_JD_PATH], [paths[2]]],
    )
    questions = [
        "第 1 个怎么样",
        "这个 JD 符合我的求职条件吗",
    ]

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=questions,
        initial_state=state,
        initial_focus=None,
        use_real_dialog_events=True,
        use_real_contextless_guard=True,
        use_real_state_updates=True,
        material_indices=[[0], [0]],
        domain_options={},
        repo_state=repo_state,
        generate=True,
    )

    assert [request.query for request in observed] == [
        questions[0],
        repo_state.docs[1],
        questions[1],
        repo_state.docs[1],
    ]
    assert captured["materials"][0]["allowed_paths"] == {_RESOLVED_JD_PATH}
    assert captured["materials"][1]["allowed_paths"] == {_RESOLVED_JD_PATH}
    assert captured["dialog_inputs"][1]["focused_file"] == _RESOLVED_JD_PATH
    assert state.last_result_set_focus_file == _RESOLVED_JD_PATH
    assert state.last_generated_result_source_hits == [
        [_RESOLVED_JD_PATH],
        [paths[2]],
    ]
    assert captured["prompts"] == []
    assert fake_models.calls == []


def test_unique_non_jd_abstains_and_preserves_normal_retrieval_fallback(
    monkeypatch,
    tmp_path,
):
    observed = _capture_recruitment_execute(monkeypatch)
    host = create_domain_host(
        plugin=RecruitmentJDPlugin(),
        expected_plugin_id=PLUGIN_ID,
    )
    path = "合同/合成服务条款.md"
    repo_state = _repo_with_documents(
        [path],
        ["合同条款：本合成文本仅约定按期交付和验收记录。"],
    )

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=["看看合成服务条款怎么样"],
        initial_state=ConversationState(),
        initial_focus=None,
        use_real_dialog_events=True,
        material_focus=path,
        material_indices=[[0]],
        domain_options=_RECRUITMENT_OPTIONS,
        repo_state=repo_state,
    )

    assert len(observed) == 2
    assert observed[1].query == repo_state.docs[0]
    assert captured["printed"] == captured["state_updates"] == ["受控本地结果"]
    assert len(captured["materials"]) == 1
    assert captured["prompts"] == []
    assert fake_models.calls == []
    assert state.last_route == "normal_retrieval"


def test_ambiguous_display_name_does_not_dispatch_a_guessed_document(
    monkeypatch,
    tmp_path,
):
    observed = _capture_recruitment_execute(monkeypatch)
    host = create_domain_host(
        plugin=RecruitmentJDPlugin(),
        expected_plugin_id=PLUGIN_ID,
    )
    paths = ["目录甲/同名资料.md", "目录乙/同名资料.md"]
    repo_state = _repo_with_documents(paths, [_LONG_SYNTHETIC_JD] * 2)

    _state, captured, _fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=["看看同名资料怎么样"],
        initial_state=ConversationState(),
        initial_focus=None,
        use_real_dialog_events=True,
        material_focus=paths[0],
        material_indices=[[0, 1]],
        domain_options=_RECRUITMENT_OPTIONS,
        repo_state=repo_state,
    )

    assert [request.query for request in observed] == ["看看同名资料怎么样"]
    assert captured["printed"] == ["受控本地结果"]
    assert len(captured["materials"]) == 1


def test_evaluation_without_resolved_file_keeps_existing_fallback(
    monkeypatch,
    tmp_path,
):
    observed = _capture_recruitment_execute(monkeypatch)
    host = create_domain_host(
        plugin=RecruitmentJDPlugin(),
        expected_plugin_id=PLUGIN_ID,
    )

    _state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=["这个方案怎么样"],
        initial_state=ConversationState(),
        initial_focus=None,
        use_real_dialog_events=True,
        material_focus=None,
        material_indices=[[]],
        domain_options=_RECRUITMENT_OPTIONS,
        repo_state=_repo_with_documents([], []),
    )

    assert [request.query for request in observed] == ["这个方案怎么样"]
    assert captured["printed"] == captured["state_updates"] == ["受控本地结果"]
    assert len(captured["materials"]) == 1
    assert captured["prompts"] == []
    assert fake_models.calls == []


@pytest.mark.parametrize(
    "options",
    [{}, {"org.example.synthetic": {"enabled": True}}],
)
def test_resolved_document_does_not_invent_missing_recruitment_rules(
    monkeypatch,
    tmp_path,
    options,
):
    observed = _capture_recruitment_execute(monkeypatch)
    host = create_domain_host(
        plugin=RecruitmentJDPlugin(),
        expected_plugin_id=PLUGIN_ID,
    )
    repo_state = _repo_with_documents(
        [_RESOLVED_JD_PATH],
        [_LONG_SYNTHETIC_JD],
    )

    _state, captured, _fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=["看看某公司A-Python岗位怎么样"],
        initial_state=ConversationState(),
        initial_focus=None,
        use_real_dialog_events=True,
        material_focus=_RESOLVED_JD_PATH,
        material_indices=[[0]],
        domain_options=options,
        repo_state=repo_state,
        generate=True,
    )

    assert observed[1].options == options
    assert captured["printed"][0].startswith("## JD 明确约束\n\n")
    assert "## 单 JD 显式规则比较" not in captured["printed"][0]


def test_direct_domain_jd_is_handled_once_without_resolved_document_retry(
    monkeypatch,
    tmp_path,
):
    observed = _capture_recruitment_execute(monkeypatch)
    host = create_domain_host(
        plugin=RecruitmentJDPlugin(),
        expected_plugin_id=PLUGIN_ID,
    )

    _state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=host,
        scripted_questions=[_LONG_SYNTHETIC_JD],
        initial_state=ConversationState(),
        initial_focus=None,
        use_real_dialog_events=True,
        domain_options=_RECRUITMENT_OPTIONS,
        repo_state=_repo_with_documents([], []),
        generate=True,
    )

    assert len(observed) == 1
    assert observed[0].query == _LONG_SYNTHETIC_JD
    assert captured["materials"] == captured["prompts"] == []
    assert captured["printed"][0].startswith("## 单 JD 显式规则比较\n\n")
    assert fake_models.calls == []
