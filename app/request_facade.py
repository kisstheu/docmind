"""One read-only request per process; business results never come from logs.

Run with the configured DocMind interpreter from its repository root:
python -m app.request_facade --request REQUEST.json --result RESULT.json
The caller owns the process timeout and private input/output locations.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


class SingleRequestPort:
    def __init__(self, question: str):
        self.question = question
        self.consumed = False
        self.result = {"protocol": 1, "ok": False, "error": "processing_failed"}

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    result = {"protocol": 1, "ok": False, "error": "initialization_failed"}
    try:
        data = json.loads(args.request.read_text(encoding="utf-8"))
        question = data["question"]
        if not isinstance(question, str) or not question.strip() or len(question) > 24000:
            raise ValueError("invalid question")
        notes = Path(data["notes_dir"]).resolve(strict=True)
        if not notes.is_dir():
            raise ValueError("invalid source scope")
        from app import chat_loop
        from app.dialog_state_machine import ConversationState
        from ask_notes import main as run

        chat_loop.conversation_state = ConversationState()
        port = SingleRequestPort(question)
        sys.argv = ["ask_notes.py", "--notes-dir", str(notes)]
        run(loop_options={"request_port": port})
        result = port.result
    except (Exception, SystemExit):
        import traceback
        traceback.print_exc()  # Private subprocess diagnostics, never the result transport.
    temp = args.result.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    temp.replace(args.result)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
