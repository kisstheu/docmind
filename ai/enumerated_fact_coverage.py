from __future__ import annotations

import os
import re
from dataclasses import dataclass


_ENABLED_ENV = "DOCMIND_ENUMERATED_FACT_COVERAGE"


@dataclass(frozen=True)
class EnumeratedFactCoverage:
    targets: tuple[str, ...]
    missing: tuple[str, ...]
    explicit_partial: bool = False

    @property
    def complete(self) -> bool:
        return not self.missing

    @property
    def safe_to_deliver(self) -> bool:
        return self.complete or self.explicit_partial


def enumerated_fact_coverage_enabled() -> bool:
    return os.getenv(_ENABLED_ENV, "1").strip().lower() not in {"0", "false", "no", "off"}


def extract_enumerated_fact_targets(question: str) -> tuple[str, ...]:
    """Extract explicitly listed value fields without using domain vocabulary."""
    if not enumerated_fact_coverage_enabled():
        return ()
    target = re.sub(r"^\s*(?:根据|按照|按|依据)[^，,。？?]+[，,]", "", question or "")
    ending = re.search(r"(?:分别|各自)?(?:是|为)?(?:多少|什么)[？?]?\s*$", target)
    if not ending:
        return ()
    target = target[:ending.start()].strip()
    if "的" in target:
        target = target.split("的", 1)[1]
    fields = tuple(
        field
        for item in re.split(r"、|以及|和|[，,]", target)
        if 2 <= len(field := item.strip()) <= 40
    )
    return fields if 2 <= len(fields) <= 12 else ()


def build_enumerated_fact_coverage_contract(question: str) -> str:
    targets = extract_enumerated_fact_targets(question)
    if not targets:
        return ""
    listed = "\n".join(f"{index}. {target}" for index, target in enumerate(targets, 1))
    return (
        "【多项事实完整性交付】\n"
        "用户明确列出的所求项如下：\n"
        f"{listed}\n"
        "按上述原顺序逐项核对参考片段。有证据的每项都必须使用对应所求项名称分别呈现；"
        "没有证据的项也必须保留并明确标为证据不足，不得静默省略。"
        "每项保留原文中的范围端点、单位、比较符号、相对基准和必要限定；"
        "事实位于相邻句段或原文未逐字使用所求项名称，都不能成为漏项理由。"
        "输出前逐项对照该清单，不得以部分命中结束回答。\n\n"
    )


def assess_enumerated_fact_coverage(question: str, answer: str) -> EnumeratedFactCoverage:
    targets = extract_enumerated_fact_targets(question)
    if not targets:
        return EnumeratedFactCoverage((), ())
    normalized_answer = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", answer or "").lower()
    missing = tuple(
        target
        for target in targets
        if re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", target).lower()
        not in normalized_answer
    )
    explicit_partial = bool(missing and re.search(
        r"(?:其他|其余|剩余)(?:请求)?项.{0,30}"
        r"(?:尚需|需要|需|证据不足|信息不足|无法确定|未找到)",
        answer or "",
    ))
    return EnumeratedFactCoverage(targets, missing, explicit_partial)


__all__ = [
    "EnumeratedFactCoverage",
    "assess_enumerated_fact_coverage",
    "build_enumerated_fact_coverage_contract",
    "enumerated_fact_coverage_enabled",
    "extract_enumerated_fact_targets",
]
