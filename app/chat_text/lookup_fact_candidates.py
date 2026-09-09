from __future__ import annotations

import os
import re

from app.chat_text.lookup_common import _normalize_lookup_token, _term_matches_line


def _quantity_attribute_slots(question: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Recognize only dimensions with an explicit, locally checkable relation."""
    if os.getenv("DOCMIND_NONLITERAL_QUANTITY_FACTS", "1") == "0":
        return "", []
    target = re.sub(r"^\s*(?:根据|按照|按|依据)[^，,。？?]+[，,]", "", question or "")
    ending = re.search(r"(?:分别|各自)?(?:是|为)?多少[？?]?\s*$", target)
    if not ending:
        return "", []
    target = target[:ending.start()].strip()
    topic = ""
    scoped = re.fullmatch(r"([^、，,和]+)的(.+)", target)
    if scoped:
        topic, target = scoped.groups()
    slots = []
    for label in re.split(r"、|以及|和|[，,]", target):
        match = re.fullmatch(r"([\u4e00-\u9fa5]{2,})(日期|标准|阈值)", label.strip())
        if match:
            anchor, dimension = match.groups()
            slots.append((label.strip(), anchor, dimension))
    return topic, slots


def _match_quantity_attribute_propositions(
    line: str, chunk: str, topic: str, slots: list[tuple[str, str, str]],
) -> dict[str, str]:
    """Return requested labels backed by a verbatim proposition, not by digits.

    A selected chunk must contain an explicit question subject when supplied.
    An event must govern a calendar date, or a numeric comparison must govern
    the requested outcome. Unknown dimensions retain the lexical fallback.
    """
    topic_norm = _normalize_lookup_token(topic)
    if topic_norm and not _term_matches_line(topic_norm, chunk.lower(), _normalize_lookup_token(chunk)):
        return {}
    # A paragraph with its own explicit scope cannot borrow the chunk's topic.
    inline_scope = re.match(r"^([^，,。；;：:]+)[：:]", line)
    if topic_norm and inline_scope and _normalize_lookup_token(inline_scope[1]) != topic_norm:
        return {}

    matches: dict[str, str] = {}
    for label, anchor, dimension in slots:
        escaped = re.escape(anchor)
        for part in re.finditer(r"[^。；;！!？?]+(?:[。；;！!？?]|$)", line):
            sentence = part.group(0).strip()
            if sentence.endswith(("?", "？")) or re.search(r"是否|能否|可否|吗|假设|假如|例如|可能", sentence):
                continue
            sentence = sentence.rstrip("。；;！!")
            if dimension == "日期":
                # A proposed or conditional date is not an asserted event date.
                if re.search(r"若|如果|假如|假设|计划|预计|可能|拟|未|不|并非", sentence):
                    continue
                date = r"\d{4}年\d{1,2}月\d{1,2}日"
                relation = re.search(rf"(?:{escaped}(?:于|在){date}|(?:于|在){date}(?:正式)?{escaped})", sentence)
            else:
                # The unit is taken from the source. No domain unit vocabulary
                # or outcome synonyms can manufacture a requested fact.
                relation = re.search(
                    r"(?:>=|<=|≥|≤|>|<|不低于|不超过|至少|至多|达到|超过|大于|小于)"
                    r"\s*\d+(?:\.\d+)?\s*"
                    r"(?:[a-zA-Z%‰℃°]+|[\u4e00-\u9fa5]{1,3}?)"
                    r"(?P<when>时|后)?(?:应|则|才|可)?"
                    r"(?:判断为|判定为|判为|认定为|视为|记为|算作)"
                    rf"(?P<outcome>[^，,。；;：:\d]*?{escaped})(?=[，,]|$)",
                    sentence,
                )
                if relation:
                    prefix = re.split(r"[，,：:]", sentence[:relation.start()])[-1]
                    if not relation['when'] and not re.search(r"(?:如果|若|如|当).+", prefix):
                        continue
                    if re.search(r"不|未|非|无", relation['outcome']) or re.search(
                        r"不应|不可|不能|不得|未能|并非", relation.group(0),
                    ):
                        continue
            if relation:
                matches[_normalize_lookup_token(label)] = relation.group(0)
                break
    return matches
