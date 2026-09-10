"""One read-only request per process; business results never come from logs.

Run with the configured DocMind interpreter from its repository root:
python -m app.request_facade --request REQUEST.json --result RESULT.json
The caller owns the process timeout and private input/output locations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys


class SingleRequestPort:
    def __init__(self, question: str):
        self.question = question
        self.consumed = False
        self.result = {"protocol": 1, "ok": False, "error": "processing_failed"}
        self.dependency = None

    def record_dependency(self, execution, candidates, question):
        from ai.evidence_dependency.user_evidence import snapshot, presentation, confirmation_execution
        original = snapshot(execution, candidates, question)
        augmented, candidates = confirmation_execution(original)
        # Targets for missing inputs can be offered on the existing gap row without
        # replacing this generation's table. Replay materializes their adopted values.
        augmented.diagnostics['row_nodes'] = execution.diagnostics['row_nodes']
        self.dependency = (snapshot(augmented, candidates, question), presentation(augmented, candidates))

    def read_question(self):
        if self.consumed:
            return "exit"
        self.consumed = True
        return self.question

    def finish_turn(self, *, state, decision_result, valid):
        if not valid or not state.last_answer_text:
            return
        self.result = {
            "protocol": 1, "ok": True,
            "answer": state.last_answer_text,
            "decision": asdict(decision_result) if decision_result is not None else None,
            "table_presentation": (
                asdict(state.current_presentation_table)
                if state.current_presentation_table is not None else None
            ),
            "source_files": list(state.last_answer_source_files or []),
        }
        if self.dependency:
            self.result['execution_snapshot'], self.result['confirmation_context'] = self.dependency


def replay(data, user_file, notes):
    from ai.evidence_dependency.user_evidence import apply_records, presentation
    from ai.evidence_dependency.delivery import deliver
    from ai.decision_result import render_decision_result
    state = data['execution_snapshot']
    manifest = state.get('source_manifest')
    if not manifest:
        raise ValueError('missing saved source manifest')
    if any(c['path'] not in manifest for c in state['candidates']):
        raise ValueError('incomplete saved source manifest')
    for name, fingerprint in manifest.items():
        if Path(name).name != name or hashlib.sha256((notes / name).read_bytes()).hexdigest() != fingerprint:
            raise ValueError('source version changed')
    records = json.loads(user_file.read_text(encoding='utf-8')) if user_file else []
    ex, candidates = apply_records(state, records)
    decision = deliver(ex, candidates, state['question'])
    return dict(protocol=1, ok=True, answer=render_decision_result(decision), decision=asdict(decision),
                table_presentation=None, source_files=list(decision.source_files), execution_snapshot=state,
                confirmation_context=presentation(ex, candidates),
                execution_diagnostics=ex.diagnostics,
                model_calls=dict(remote_generation=0, local_verification=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--user-evidence", type=Path)
    args = parser.parse_args()
    result = {"protocol": 1, "ok": False, "error": "initialization_failed"}
    try:
        data = json.loads(args.request.read_text(encoding="utf-8"))
        if set(data) - {'question', 'notes_dir', 'operation', 'execution_snapshot'}:
            raise ValueError('unsupported request fields')
        question = data["question"]
        if not isinstance(question, str) or not question.strip() or len(question) > 24000:
            raise ValueError("invalid question")
        notes = Path(data["notes_dir"]).resolve(strict=True)
        if not notes.is_dir():
            raise ValueError("invalid source scope")
        if data.get('operation') == 'confirm':
            result = replay(data, args.user_evidence, notes)
        else:
            if args.user_evidence or data.get('execution_snapshot') or data.get('operation') not in (None, 'generate'):
                raise ValueError('user evidence requires saved execution')
            from app import chat_loop
            from app.dialog_state_machine import ConversationState
            from ask_notes import main as run

            chat_loop.conversation_state = ConversationState()
            port = SingleRequestPort(question)
            sys.argv = ["ask_notes.py", "--notes-dir", str(notes)]
            run(loop_options={"request_port": port})
            result = port.result
            if result.get('execution_snapshot'):
                result['execution_snapshot']['source_manifest'] = {
                    name: hashlib.sha256((notes / name).read_bytes()).hexdigest()
                    for name in dict.fromkeys(c['path'] for c in result['execution_snapshot']['candidates'])
                    if Path(name).name == name}
    except (Exception, SystemExit):
        import traceback
        traceback.print_exc()  # Private subprocess diagnostics, never the result transport.
    temp = args.result.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temp.replace(args.result)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
