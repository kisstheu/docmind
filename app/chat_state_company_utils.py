from __future__ import annotations

import re

# Keep the list minimal and generic; these are common non-entity references.
_GENERIC_COMPANY_REFERENCES = {
    "\u516c\u53f8",  # 公司
    "\u8be5\u516c\u53f8",  # 该公司
    "\u672c\u516c\u53f8",  # 本公司
    "\u67d0\u516c\u53f8",  # 某公司
    "\u4f01\u4e1a",  # 企业
    "\u5355\u4f4d",  # 单位
    "\u7ec4\u7ec7",  # 组织
    "\u540d\u79f0",  # 名称
    "\u5408\u4f5c\u4f19\u4f34",  # 合作伙伴
    "\u5408\u4f5c\u65b9",  # 合作方
}

_STRONG_COMPANY_MARKERS = (
    "\u6709\u9650\u516c\u53f8",  # 有限公司
    "\u80a1\u4efd\u6709\u9650\u516c\u53f8",  # 股份有限公司
    "\u96c6\u56e2",  # 集团
    "\u79d1\u6280",  # 科技
    "\u5b9e\u4e1a",  # 实业
)


def normalize_company_item(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # Drop explanatory suffixes that are not part of the company name.
    t = re.sub(r"[\(\uff08](?:full name not given|alias|placeholder.*?)[\)\uff09]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"(?:\.\.\.|\u2026)+$", "", t).strip()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def is_generic_company_reference(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True

    t_lower = t.lower()
    if t in _GENERIC_COMPANY_REFERENCES or t_lower in _GENERIC_COMPANY_REFERENCES:
        return True

    if re.fullmatch(r"[\u67d0\u8be5\u672c\u6b64\u8fd9\u90a3\u4e00]\S*\u516c\u53f8", t):
        return True
    if re.fullmatch(r"[\u4e00-\u9fa5]{1,6}\u516c\u53f8", t) and len(t) <= 4:
        return True

    bad_fragments = (
        "\u53c2\u8003",  # 参考
        "\u6839\u636e",  # 根据
        "\u5982\u4e0b",  # 如下
        "\u4ee5\u4e0b",  # 以下
        "\u672a\u627e\u5230",  # 未找到
    )
    return any(x in t for x in bad_fragments)


def looks_like_real_company_name(text: str) -> bool:
    t = (text or "").strip()
    if not t or is_generic_company_reference(t):
        return False

    t_lower = t.lower()
    if len(t) < 3:
        return False

    if re.fullmatch(r"[\u4e00-\u9fa5]{2,6}\u516c\u53f8", t):
        return False
    if re.fullmatch(r"[\u67d0\u8be5\u672c\u6b64\u8fd9\u90a3\u4e00][\u4e00-\u9fa5]*\u516c\u53f8", t):
        return False

    if any(marker in t for marker in _STRONG_COMPANY_MARKERS):
        return True

    if re.search(r"\b(co|company|inc|ltd|corp|group|holdings?|technology|tech)\b", t_lower):
        return True

    if re.search(r"[a-z]", t_lower):
        parts = [x for x in re.split(r"[^a-z0-9']+", t_lower) if x]
        if len(parts) >= 2 and len("".join(parts)) >= 8:
            return True

    if re.fullmatch(r"[\u4e00-\u9fa5]{3,12}", t):
        return True

    return False
