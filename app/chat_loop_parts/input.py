from __future__ import annotations

import importlib
import os
import sys
import time


def _try_enable_posix_readline(*, os_name=None, import_module=None) -> bool:
    current_os_name = os.name if os_name is None else os_name
    if current_os_name != "posix":
        return False

    importer = importlib.import_module if import_module is None else import_module
    try:
        importer("readline")
    except Exception:
        return False
    return True


_POSIX_READLINE_ENABLED = _try_enable_posix_readline()


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


def _read_fresh_tty_line(prompt: str = "\n问：", *, input_func=None) -> str:
    read_input = input if input_func is None else input_func
    return read_input(prompt)


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
    input_func=None,
    tty_input_func=None,
    should_use_fresh_tty_input=_should_use_fresh_tty_input,
    has_buffered_input=_has_buffered_console_input,
    max_buffered_lines: int = 4,
    debounce_seconds: float = 0.12,
    sleep_func=time.sleep,
    monotonic_func=time.monotonic,
    use_fresh_tty_input: bool = False,
) -> str:
    read_input = input if input_func is None else input_func
    if use_fresh_tty_input and should_use_fresh_tty_input():
        if tty_input_func is None:
            line = _read_fresh_tty_line(prompt, input_func=read_input)
        else:
            line = tty_input_func(prompt)
        return _merge_user_question_lines([line])

    lines = [read_input(prompt)]
    if max_buffered_lines <= 1:
        return _merge_user_question_lines(lines)

    wait_for_more = not _merge_user_question_lines(lines)
    deadline = monotonic_func() + max(debounce_seconds, 0.0)
    while len(lines) < max_buffered_lines:
        if has_buffered_input():
            lines.append(read_input(""))
            wait_for_more = not _merge_user_question_lines(lines)
            deadline = monotonic_func() + min(max(debounce_seconds, 0.0), 0.05)
            continue
        if not wait_for_more or monotonic_func() >= deadline:
            break
        sleep_func(0.02)

    return _merge_user_question_lines(lines)
