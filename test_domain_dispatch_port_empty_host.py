from __future__ import annotations

import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

import app.chat_loop as runtime
import app.chat_loop_parts.runner as runner
import ask_notes
from app.dialog_state_machine import ConversationState, DialogEvent
from app.domain_dispatch_port import DomainDispatchPort
from app.domain_host import EmptyDomainHost
from bootstrap.domain_composition import create_domain_host
from docmind_domain_sdk import DomainRequest


class _Logger:
    def debug(self, *_args, **_kwargs):
        return None

    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def error(self, *_args, **_kwargs):
        return None


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


def _run_turn(monkeypatch, tmp_path, *, port, gate=None, generate=False):
    question = "哪些文档里提到了检索策略？"
    state = ConversationState(
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
    questions = iter([question, "q"])
    captured = {
        "search": [],
        "materials": [],
        "printed": [],
        "state_updates": [],
    }
    fake_models = _FakeModels()

    monkeypatch.setattr(runtime, "build_chat_config", lambda _repo: {"stable": True})
    monkeypatch.setattr(runtime, "_flush_pending_tty_input_unix", lambda: None)
    monkeypatch.setattr(
        runtime,
        "_read_user_question",
        lambda **_kwargs: next(questions),
    )
    monkeypatch.setattr(
        runner,
        "handle_file_action_turn",
        lambda **kwargs: (
            gate == "file_action",
            kwargs["state"],
            "focus.md",
        ),
    )
    monkeypatch.setattr(
        runner,
        "detect_dialog_event",
        lambda *_args, **_kwargs: DialogEvent(name="unknown"),
    )
    monkeypatch.setattr(runner, "apply_event_to_state", lambda current, _event: current)
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
    monkeypatch.setattr(
        runner,
        "resolve_route",
        lambda *_args, **_kwargs: {
            "route": route,
            "smalltalk_reply": "",
            "route_question_input": question,
        },
    )
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
        return {
            "current_focus_file": "focus.md",
            "relevant_indices": [2, 0],
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
    monkeypatch.setattr(runner, "looks_like_focused_file_content_question", lambda _q: False)
    monkeypatch.setattr(runner, "maybe_build_direct_lookup_answer", lambda **_kwargs: None)
    monkeypatch.setattr(
        runtime,
        "try_handle_retrieval_force_local_or_empty_context",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(runner, "build_safe_final_prompt", lambda **_kwargs: "受控提示")
    monkeypatch.setattr(
        runner,
        "print_answer",
        lambda answer, _start: captured["printed"].append(answer),
    )
    monkeypatch.setattr(runner, "append_memory", lambda *_args: None)

    def fake_update(current, _question, answer, _logger, **_kwargs):
        captured["state_updates"].append(answer)
        return current

    monkeypatch.setattr(runner, "update_state_after_retrieval_answer", fake_update)
    runner.run_chat_loop(
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
    return state, captured, fake_models


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
    monkeypatch.setattr(ask_notes, "create_domain_host", lambda: host)
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
