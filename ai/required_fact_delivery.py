from __future__ import annotations

import os
import re


_ENABLED_ENV = "DOCMIND_REQUIRED_FACT_DELIVERY"
_MULTI_PART_REQUEST_RE = re.compile(r"(?:分别|各自).*(?:什么|哪些|怎样|如何|多少|要求|限制|条件)")


def required_fact_delivery_enabled() -> bool:
    return os.getenv(_ENABLED_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def needs_required_fact_delivery(question: str) -> bool:
    """Detect multi-part requests without using vocabulary from a business domain."""
    if not required_fact_delivery_enabled():
        return False
    normalized = re.sub(r"\s+", "", question or "")
    return bool(_MULTI_PART_REQUEST_RE.search(normalized))


def build_required_fact_delivery_contract(question: str) -> str:
    if not needs_required_fact_delivery(question):
        return ""
    return (
        "【共同适用必要事实完整性交付】\n"
        "用户正在同一问题中逐项询问多个对象。除逐项核对并回答外，还要检查参考片段中"
        "与这些对象处于同一适用范围的共同限定。凡会改变已列项结论如何理解或执行的"
        "共同条件、时序、例外、前置要求或后续动作，都是回答所必需的事实，必须单独呈现；"
        "不能因为它不对应某一个所求项名称而省略。原文把多个对象与同一结论、条件或理由"
        "共同绑定时，拆分呈现不得只把共享事实保留在其中一项；每个适用对象都要保持绑定，"
        "也可以合并这些对象后一次完整说明。只补充与所问对象直接相关且证据明确的"
        "共同事实，不扩展到旁支主题。仅在同段出现、但不改变已列项结论的背景、相邻操作细节"
        "或扩展说明不属于必要事实，不要补充。保留原文的义务强度，不把“应”或“可”改写为"
        "“必须”。输出前同时核对逐项事实与共同适用的必要事实。\n\n"
    )


__all__ = [
    "build_required_fact_delivery_contract",
    "needs_required_fact_delivery",
    "required_fact_delivery_enabled",
]
