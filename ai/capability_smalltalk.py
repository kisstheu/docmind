from __future__ import annotations

from ai.capability_common import EMOJI_RESPONSES, JUDGMENT_LIKE_KEYWORDS, clean_text, contains_any


def answer_smalltalk(question: str, dialog_state=None) -> str | None:
    q = clean_text(question)

    if contains_any(q, JUDGMENT_LIKE_KEYWORDS):
        return None

    if contains_any(q, ("你好", "嗨", "hello", "hi")):
        return "你好呀。"

    if contains_any(q, ("在吗", "在不在", "在嘛")):
        return "在。"

    if contains_any(q, ("谢谢", "多谢", "感谢", "辛苦了")):
        return "不客气。"

    if contains_any(q, ("不少", "挺多", "很多")):
        if dialog_state and getattr(dialog_state, "last_route", None) == "repo_meta":
            if getattr(dialog_state, "last_local_topic", None) == "list_files":
                return "是的，现在已经积累得挺多了。"
        return "嗯，确实不少。"

    if contains_any(q, ("太细了", "太碎了", "别那么细", "不要太细")):
        if dialog_state and getattr(dialog_state, "last_route", None) == "repo_meta":
            if getattr(dialog_state, "last_local_topic", None) in {"category", "category_summary"}:
                return "嗯，我按更大的方面再给你归并一下。"
        return "嗯，我换个更粗的角度说。"

    if q in EMOJI_RESPONSES:
        if dialog_state and getattr(dialog_state, "last_route", None) == "system_capability":
            return "先别急着夸，真问起来我再表现。"
        return "哈哈。"

    if contains_any(q, ("挺强", "真强", "厉害", "牛", "不错", "可以啊", "真行", "这么强", "看着挺强")):
        if dialog_state and getattr(dialog_state, "last_route", None) == "system_capability":
            return "功能先说在前面，真干活还得看你怎么使唤我。"
        return "你这么一说，我都有点不好意思了。"

    if contains_any(q, ("不谦虚", "挺自信", "还真敢讲", "你还挺会说")):
        return "先把活干明白再谦虚，也不迟。"

    return None
