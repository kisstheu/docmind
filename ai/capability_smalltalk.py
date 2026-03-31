from __future__ import annotations

import os
from datetime import datetime

from ai.capability_common import JUDGMENT_LIKE_KEYWORDS, clean_text, contains_any


def _emoji_enabled() -> bool:
    raw = (os.getenv("DOCMIND_EMOJI_ENABLED") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _emoji_level() -> int:
    raw = (os.getenv("DOCMIND_EMOJI_LEVEL") or "1").strip()
    try:
        level = int(raw)
    except ValueError:
        level = 1
    return max(0, min(level, 2))


def _with_emojis(text: str, *emojis: str) -> str:
    if not _emoji_enabled():
        return text

    level = _emoji_level()
    if level <= 0:
        return text

    unique_emojis: list[str] = []
    for emoji in emojis:
        e = (emoji or "").strip()
        if e and e not in unique_emojis:
            unique_emojis.append(e)
    if not unique_emojis:
        return text

    count = 1 if level == 1 else 2
    suffix = " ".join(unique_emojis[:count])
    return f"{text} {suffix}"


def answer_smalltalk(question: str, dialog_state=None) -> str | None:
    q = clean_text(question)

    if contains_any(q, JUDGMENT_LIKE_KEYWORDS):
        return None

    if contains_any(q, ("你叫什么", "你叫啥", "你叫啥名", "你叫啥名字", "你的名字", "你的姓名", "what is your name")):
        return "你可以叫我 DocMind。"

    if contains_any(q, ("你多大", "你几岁", "你多少岁", "你今年多大", "你的年龄", "你的年纪", "how old are you")):
        return "我是 AI 助手，没有真实年龄。"

    if contains_any(question, ("几点了", "现在几点", "现在几点了")):
        now = datetime.now()
        return _with_emojis(f"现在时间是 {now.strftime('%H:%M')}。", "⏰")

    return None
