from __future__ import annotations

import os

from ai.capability_common import EMOJI_RESPONSES, JUDGMENT_LIKE_KEYWORDS, clean_text, contains_any


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

    if contains_any(q, ("你好", "嗨", "hello", "hi")):
        return _with_emojis("你好呀。", "😊")

    if contains_any(q, ("在吗", "在不在", "在嘛")):
        return _with_emojis("在。", "👋")

    if contains_any(q, ("谢谢", "多谢", "感谢", "辛苦了")):
        return _with_emojis("不客气。", "🙏", "😊")

    if contains_any(q, ("不少", "挺多", "很多")):
        if dialog_state and getattr(dialog_state, "last_route", None) == "repo_meta":
            if getattr(dialog_state, "last_local_topic", None) == "list_files":
                return _with_emojis("是的，现在已经积累得挺多了。", "🙂")
        return _with_emojis("嗯，确实不少。", "🙂")

    if contains_any(q, ("太细了", "太碎了", "别那么细", "不要太细")):
        if dialog_state and getattr(dialog_state, "last_route", None) == "repo_meta":
            if getattr(dialog_state, "last_local_topic", None) in {"category", "category_summary"}:
                return _with_emojis("嗯，我按更大的方面再给你归并一下。", "👌")
        return _with_emojis("嗯，我换个更粗的角度说。", "👌")

    if q in EMOJI_RESPONSES:
        if dialog_state and getattr(dialog_state, "last_route", None) == "system_capability":
            return _with_emojis("先别急着夸，真问起来我再表现。", "😄")
        return _with_emojis("哈哈。", "😄")

    if contains_any(q, ("挺强", "真强", "厉害", "牛", "不错", "可以啊", "真行", "这么强", "看着挺强")):
        if dialog_state and getattr(dialog_state, "last_route", None) == "system_capability":
            return _with_emojis("功能先说在前面，真干活还得看你怎么使唤我。", "😄")
        return _with_emojis("你这么一说，我都有点不好意思了。", "😄", "🙌")

    if contains_any(q, ("不谦虚", "挺自信", "还真敢讲", "你还挺会说")):
        return _with_emojis("先把活干明白再谦虚，也不迟。", "😄")

    return None
