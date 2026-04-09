from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_ENABLED_TRUE_VALUES = {"1", "true", "yes", "on"}


def _is_enabled() -> bool:
    raw = (os.getenv("DOCMIND_DEBUG_SAVE_QUESTIONS") or "").strip().lower()
    return raw in _ENABLED_TRUE_VALUES


def _safe_slug(text: str) -> str:
    value = (text or "").strip() or "notes"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value)


def _resolve_debug_root() -> Path:
    raw = (os.getenv("DOCMIND_DEBUG_SAVE_DIR") or "").strip()
    if not raw:
        return Path("logs") / "debug_questions"
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    return path


@dataclass
class DebugQuestionRecorder:
    session_file: Path
    last_run_file: Path
    notes_dir: Path
    _counter: int = field(default=0, init=False)
    _header: str = field(default="", init=False)
    _last_run_initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        self.last_run_file.parent.mkdir(parents=True, exist_ok=True)

        started_at = datetime.now().isoformat(timespec="seconds")
        self._header = (
            f"# session_started: {started_at}\n"
            f"# notes_dir: {self.notes_dir}\n\n"
        )
        self.session_file.write_text(self._header, encoding="utf-8")

    def record(self, question: str, normalized_question: str | None = None) -> None:
        raw_question = (question or "").strip()
        if not raw_question:
            return

        normalized = (normalized_question or "").strip() or raw_question

        self._counter += 1
        payload = {
            "index": self._counter,
            "time": datetime.now().isoformat(timespec="seconds"),
            "question": normalized,
        }
        if normalized != raw_question:
            payload["raw_question"] = raw_question

        line = json.dumps(payload, ensure_ascii=False) + "\n"
        self._append(self.session_file, line)
        self._ensure_last_run_ready()
        self._append(self.last_run_file, line)

    @staticmethod
    def _append(path: Path, text: str) -> None:
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            # Debug recorder must never affect main flow.
            return

    def _ensure_last_run_ready(self) -> None:
        if self._last_run_initialized:
            return
        try:
            self.last_run_file.write_text(self._header, encoding="utf-8")
            self._last_run_initialized = True
        except Exception:
            # Debug recorder must never affect main flow.
            return


def build_debug_question_recorder(notes_dir: Path, logger=None) -> DebugQuestionRecorder | None:
    if not _is_enabled():
        return None

    root = _resolve_debug_root()
    run_dir = root / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = run_dir / f"questions_{stamp}_{_safe_slug(notes_dir.name)}.log"
    last_run_file = root / "last_run_questions.log"

    recorder = DebugQuestionRecorder(
        session_file=session_file,
        last_run_file=last_run_file,
        notes_dir=notes_dir.resolve(),
    )

    if logger is not None:
        logger.info(f"[debug] question trace enabled: {recorder.last_run_file}")

    return recorder
