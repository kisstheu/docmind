from __future__ import annotations

import re

EXPANSION_MARKERS = ["\u66f4\u591a", "\u8fd8\u6709", "\u7ee7\u7eed", "\u518d\u6765", "\u8865\u5145"]
NO_NEW_PATTERNS = [
    "\u6ca1\u6709\u53d1\u73b0\u5176\u4ed6",
    "\u6ca1\u6709\u53d1\u73b0\u65b0\u7684",
    "\u6ca1\u6709\u5176\u4ed6\u65b0\u7684",
    "\u6682\u672a\u53d1\u73b0\u65b0\u7684",
    "\u672a\u53d1\u73b0\u5176\u4ed6",
    "\u6ca1\u6709\u627e\u5230\u5176\u4ed6",
    "\u6ca1\u6709\u627e\u5230\u65b0\u7684",
    "\u672a\u627e\u5230\u65b0\u7684",
    "\u6ca1\u6709\u66f4\u591a",
    "\u6682\u65e0\u66f4\u591a",
    "\u6ca1\u6709\u8bc6\u522b\u51fa\u65b0\u7684",
    "\u672a\u8bc6\u522b\u51fa\u65b0\u7684",
]

_COMPANY_WORD = "\u516c\u53f8"
_NAME_WORD = "\u540d\u79f0"
_FILE_WORDS = ("\u6587\u4ef6", "\u6587\u6863", "\u8bb0\u5f55")
_PERSON_WORDS = ("\u4eba\u7269", "\u4eba\u5458", "\u4eba\u540d", "\u59d3\u540d")
_FOLLOWUP_HINTS = ("\u5417", "\u54ea\u4e2a", "\u54ea", "\u5bf9\u5e94", "\u8fd9", "\u90a3", "\u524d\u9762", "\u4e0a\u9762")


def _contains_no_new_signal(answer_text: str) -> bool:
    a = (answer_text or "").strip()
    if not a:
        return False
    return any(p in a for p in NO_NEW_PATTERNS)


def _looks_like_file_locator_answer(answer_text: str) -> bool:
    a = (answer_text or "").strip()
    if not a:
        return False
    if re.search(r"(?:\u6587\u4ef6|\u6587\u6863|\u8bb0\u5f55)[\u3010\[][^\]\u3011\n]+[\u3011\]]", a):
        return True
    if re.search(r"(?:\u5728|\u4f4d\u4e8e).{0,8}(?:\u6587\u4ef6|\u6587\u6863|\u8bb0\u5f55)", a):
        return True
    return False


def _has_company_list_intent(question: str) -> bool:
    if _COMPANY_WORD not in question:
        return False
    intent_tokens = (
        "\u591a\u5c11",
        "\u54ea\u4e9b",
        "\u54ea\u51e0\u5bb6",
        "\u54ea\u51e0\u4e2a",
        "\u54ea\u4e2a",
        "\u5217\u51fa",
        "\u5217\u4e00\u4e0b",
        "\u63d0\u5230",
    )
    return any(x in question for x in intent_tokens)


def infer_answer_type(user_question: str, answer_text: str) -> str | None:
    q = (user_question or "").strip()
    q_norm = re.sub(r"[\s\u3002\uff1f\uff01\uff0c,.;；]+", "", q)
    a = (answer_text or "").strip()
    numbered_lines = re.findall(r"(?:^|\n)\s*(?:\d+[.\u3001]|[-*•])\s*", a)

    if any(x in q for x in ("\u516c\u53f8\u540d", "\u516c\u53f8\u540d\u79f0", "\u4f01\u4e1a\u540d\u79f0", "\u5355\u4f4d\u540d\u79f0", "\u7ec4\u7ec7\u540d\u79f0")):
        if len(numbered_lines) >= 2:
            return "enumeration_company"

    if _has_company_list_intent(q):
        if len(numbered_lines) >= 2:
            return "enumeration_company"
        if "\u4ee5\u4e0b\u516c\u53f8" in a:
            return "enumeration_company"
        if "\u63d0\u5230\u4e86\u4ee5\u4e0b\u516c\u53f8" in a:
            return "enumeration_company"
        if ("\u603b\u5171" in a and _COMPANY_WORD in a) or a.count("\u6709\u9650\u516c\u53f8") >= 2:
            return "enumeration_company"

    if (
        len(numbered_lines) >= 1
        and re.search(r"(?:^|\n)\s*\d+[.\u3001]\s*.+?(?:HR|hr)\s*[:\uff1a]", a, flags=re.IGNORECASE)
        and (
            re.search(r"(?:\u6709\u9650\u516c\u53f8|\u80a1\u4efd\u6709\u9650\u516c\u53f8|\u96c6\u56e2|\u79d1\u6280|\u8f6f\u4ef6|\u4fe1\u606f)", a)
            or re.search(r"\u516c\u53f8[-\s]*HR", a, flags=re.IGNORECASE)
        )
    ):
        return "enumeration_company"

    if any(x in q_norm for x in EXPANSION_MARKERS):
        if len(numbered_lines) >= 1 and (_COMPANY_WORD in a or _NAME_WORD in a):
            return "enumeration_company"
        if len(numbered_lines) >= 1 and any(x in a for x in _PERSON_WORDS):
            return "enumeration_person"
        if len(numbered_lines) >= 1 and any(x in a for x in _FILE_WORDS):
            return "enumeration_file"

        if (_COMPANY_WORD in a or _NAME_WORD in a) and _contains_no_new_signal(a):
            return "enumeration_company"
        if any(x in a for x in _PERSON_WORDS) and _contains_no_new_signal(a):
            return "enumeration_person"
        if any(x in a for x in _FILE_WORDS) and _contains_no_new_signal(a):
            return "enumeration_file"

    if any(x in q for x in _FILE_WORDS):
        if len(numbered_lines) >= 2:
            return "enumeration_file"
        if "\u4ee5\u4e0b\u6587\u4ef6" in a or "\u4ee5\u4e0b\u6587\u6863" in a:
            return "enumeration_file"
        if re.search(r"(?:\u6587\u4ef6|\u6587\u6863|\u8bb0\u5f55)[\u3010\[][^\]\u3011\n]+[\u3011\]]", a):
            return "enumeration_file"

    if ("\u8c01" in q) or ("\u54ea\u4e9b\u4eba" in q) or ("\u4eba\u7269" in q):
        if len(numbered_lines) >= 2:
            return "enumeration_person"
        if "\u4ee5\u4e0b\u4eba\u7269" in a or "\u4ee5\u4e0b\u4eba\u5458" in a:
            return "enumeration_person"

    file_items = extract_file_items(a)
    if file_items and _looks_like_file_locator_answer(a):
        if any(x in q for x in _FOLLOWUP_HINTS) or any(x in q for x in _FILE_WORDS):
            return "enumeration_file"

    return None


def extract_numbered_items(answer_text: str) -> list[str]:
    items: list[str] = []

    for line in (answer_text or "").splitlines():
        line = line.strip()
        if not line:
            continue

        m = re.match(r"^\d+[.\u3001]\s*(.+?)\s*$", line)
        if m:
            items.append(m.group(1).strip())
            continue

        m = re.match(r"^[-*•]\s*(.+?)\s*$", line)
        if m:
            items.append(m.group(1).strip())
            continue

    return items


def extract_file_items(answer_text: str) -> list[str]:
    items: list[str] = []
    file_pattern = re.compile(
        r"([A-Za-z0-9_\-\u4e00-\u9fa5\\/:.\s]+?\.(?:txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx|png|jpg|jpeg|bmp|webp))\b",
        flags=re.IGNORECASE,
    )

    def _extract_file_candidate(text: str) -> str:
        raw = (text or "").strip()
        if not raw:
            return ""
        m = file_pattern.search(raw)
        if not m:
            return ""
        candidate = m.group(1).strip()
        candidate = re.sub(r"\s+", " ", candidate).strip()
        candidate = candidate.strip("\"'[]\u3010\u3011\uff08\uff09() \u3002\uff1b;,\uff0c")
        return candidate

    def _add_file_item(raw_item: str) -> None:
        candidate = _extract_file_candidate(raw_item)
        if candidate and candidate not in items:
            items.append(candidate)

    for item in extract_numbered_items(answer_text):
        _add_file_item(item)

    for m in re.findall(r"(?:\u6587\u4ef6|\u6587\u6863|\u8bb0\u5f55)[\u3010\[]([^\]\u3011\n]+)[\u3011\]]", answer_text or ""):
        _add_file_item(m)

    for m in re.findall(r"(?:\u6765\u6e90|\u51fa\u5904|(?:\u539f)?\u6587\u4ef6)\s*[:\uff1a]\s*([^\n]+)", answer_text or "", flags=re.IGNORECASE):
        _add_file_item(m)

    return items


def infer_local_answer_type(user_question: str, answer_text: str, local_topic: str | None) -> str | None:
    q = (user_question or "").strip()
    a = (answer_text or "").strip()
    topic = (local_topic or "").strip()

    if topic == "time" and any(x in q for x in _FILE_WORDS):
        if extract_file_items(a):
            return "enumeration_file"

    if topic in {"list_files", "list_files_by_topic"}:
        if extract_file_items(a):
            return "enumeration_file"

    return None

