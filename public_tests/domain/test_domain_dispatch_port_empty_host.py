from __future__ import annotations

import asyncio
import gc
import inspect
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
    PluginError,
)


_PLUGIN_ID = "org.example.neutral"


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
    material_indices=None,
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
            "current_focus_file": kwargs["current_focus_file"],
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

    def fake_update(current, _question, answer, _logger, **_kwargs):
        captured["state_updates"].append(answer)
        return current

    monkeypatch.setattr(runner, "update_state_after_retrieval_answer", fake_update)
    captured["runner_return"] = runner.run_chat_loop(
        SimpleNamespace(),
        None,
        SimpleNamespace(models=fake_models),
        "model-id",
        "http://local.invalid",
        "local-model",
        _Logger(),
        notes_dir=tmp_path,
        change_log_file=tmp_path / "changes.jsonl",
        domain_dispatch_port=port,
    )
    return runtime.conversation_state, captured, fake_models


def test_empty_host_satisfies_sync_port_and_has_no_side_effects(capsys):
    host = EmptyDomainHost()
    request = DomainRequest(request_id="request-1", query="原始问题", source_scope=())

    assert isinstance(host, DomainDispatchPort)
    assert not inspect.iscoroutinefunction(host.dispatch)
    assert host.dispatch(request) is None
    assert request.query == "原始问题"
    assert request.source_scope == ()
    assert capsys.readouterr() == ("", "")
    assert isinstance(create_domain_host(), EmptyDomainHost)


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
    monkeypatch.setattr(ask_notes, "_parse_args", lambda: SimpleNamespace(notes_dir=None))
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
    assert isinstance(captured["kwargs"]["domain_dispatch_port"], DomainDispatchPort)


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
    "gate",
    ["file_action", "contextless", "system_capability", "repo_meta", "smalltalk", "out_of_scope"],
)
def test_common_guards_do_not_dispatch(monkeypatch, tmp_path, gate):
    state = ConversationState()
    port = _SpyPort(state)

    _run_turn(monkeypatch, tmp_path, port=port, gate=gate)

    assert port.requests == []


def test_empty_host_reaches_controlled_generation_boundary(monkeypatch, tmp_path):
    state = ConversationState()
    port = _SpyPort(state)

    state, captured, fake_models = _run_turn(
        monkeypatch,
        tmp_path,
        port=port,
        generate=True,
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
    assert state.last_route == "normal_retrieval"


def test_static_minimal_handled_result_is_presented_once_and_short_circuits_retrieval(
    monkeypatch,
    tmp_path,
    capsys,
):
    plugin = _NeutralPlugin(_domain_result)
    host = create_domain_host(plugin=plugin, expected_plugin_id=_PLUGIN_ID)

    state, captured, fake_models = _run_turn(monkeypatch, tmp_path, port=host)
    output = capsys.readouterr()

    assert len(plugin.execute_requests) == 1
    request = plugin.execute_requests[0]
    assert isinstance(request, DomainRequest)
    assert request.query == "哪些文档里提到了检索策略？"
    assert request.source_scope == ()
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

    state, captured, fake_models = _run_turn(monkeypatch, tmp_path, port=host)

    assert len(plugin.execute_requests) == 1
    assert captured["printed"] == captured["state_updates"] == ["受控本地结果"]
    assert len(captured["search"]) == len(captured["materials"]) == 1
    assert captured["prompts"] == []
    assert fake_models.calls == []
    assert state.last_route == "normal_retrieval"
    assert state.last_result_set_items == ["scope-a.md", "scope-b.md"]
    assert state.last_selected_candidate == "候选项X"


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

    state, captured, fake_models = _run_turn(monkeypatch, tmp_path, port=port)

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
