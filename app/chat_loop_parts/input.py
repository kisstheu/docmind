from __future__ import annotations

import os
import sys
import time


def _has_buffered_console_input() -> bool:
    if os.name != "nt" or not sys.stdin.isatty():
        return False

    try:
        import msvcrt
    except Exception:
        return False

    try:
        return bool(msvcrt.kbhit())
    except Exception:
        return False


def _should_use_fresh_tty_input() -> bool:
    return os.name == "posix" and bool(getattr(sys.stdin, "isatty", lambda: False)())


def _flush_pending_tty_input_unix() -> bool:
    if not _should_use_fresh_tty_input():
        return False

    try:
        import termios

        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        return True
    except Exception:
        return False


def _read_fresh_tty_line(prompt: str = "\n问：") -> str:
    try:
        print(prompt, end="", flush=True)
        return sys.stdin.readline()
    except Exception:
        return input(prompt)


def _merge_user_question_lines(lines: list[str]) -> str:
    parts: list[str] = []
    for raw in lines:
        text = " ".join(str(raw or "").split())
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _read_user_question(
    prompt: str = "\n问：",
    *,
    input_func=input,
    tty_input_func=_read_fresh_tty_line,
    should_use_fresh_tty_input=_should_use_fresh_tty_input,
    has_buffered_input=_has_buffered_console_input,
    max_buffered_lines: int = 4,
    debounce_seconds: float = 0.12,
    sleep_func=time.sleep,
    monotonic_func=time.monotonic,
    use_fresh_tty_input: bool = False,
) -> str:
    if use_fresh_tty_input and should_use_fresh_tty_input():
        return _merge_user_question_lines([tty_input_func(prompt)])

    lines = [input_func(prompt)]
    if max_buffered_lines <= 1:
        return _merge_user_question_lines(lines)

    wait_for_more = not _merge_user_question_lines(lines)
    deadline = monotonic_func() + max(debounce_seconds, 0.0)
    while len(lines) < max_buffered_lines:
        if has_buffered_input():
            lines.append(input_func(""))
            wait_for_more = not _merge_user_question_lines(lines)
            deadline = monotonic_func() + min(max(debounce_seconds, 0.0), 0.05)
            continue
        if not wait_for_more or monotonic_func() >= deadline:
            break
        sleep_func(0.02)

    return _merge_user_question_lines(lines)
