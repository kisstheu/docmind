from __future__ import annotations

import re


_FILE_EXT_PATTERN = re.compile(
    r"([A-Za-z0-9_\-\u4e00-\u9fa5\s]+?\.(?:txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx|png|jpg|jpeg|bmp|webp))",
    flags=re.IGNORECASE,
)
_POLITE_PREFIX_PATTERN = re.compile(
    r"^(?:给我|帮我|请|麻烦|看下|看看|查看|打开|读下|读一个|查下|查一个|展示一下|please|open|show)+",
    flags=re.IGNORECASE,
)


def extract_explicit_file_anchors(text: str) -> list[str]:
    q = (text or "").strip()
    if not q:
        return []

    anchors: list[str] = []
    for match in _FILE_EXT_PATTERN.findall(q):
        raw = re.sub(r"\s+", "", (match or "").strip())
        raw = _POLITE_PREFIX_PATTERN.sub("", raw)
        raw = raw.strip("，。！？；:!?;:[]【】（）()\"'`")
        if not raw:
            continue

        stem = re.sub(
            r"\.(txt|md|pdf|doc|docx|xls|xlsx|csv|ppt|pptx|png|jpg|jpeg|bmp|webp)$",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        for token in (raw, stem):
            t = (token or "").strip()
            if t and t not in anchors:
                anchors.append(t)

    return anchors


def force_append_anchor_terms(query: str, anchors: list[str], logger=None) -> str:
    q = (query or "").strip()
    if not anchors:
        return q

    merged_terms = [x for x in q.split() if x.strip()]
    appended = []
    for anchor in anchors:
        if anchor not in merged_terms:
            merged_terms.append(anchor)
            appended.append(anchor)

    if appended and logger:
        logger.info(f"🧷 [文件锚词回补] {appended}")

    return " ".join(merged_terms).strip()


def extract_selector_anchors(text: str) -> list[str]:
    q = (text or "").strip()
    if not q:
        return []

    anchors: list[str] = []

    for raw in re.findall(r"\d{1,2}\s*[-–—~]\s*\d{1,2}\s*[kK]", q):
        t = re.sub(r"\s+", "", raw)
        t = re.sub(r"[-–—~]", "-", t)
        if t and t not in anchors:
            anchors.append(t)

    for raw in re.findall(r"\d{1,2}\s*\u85aa", q):
        t = re.sub(r"\s+", "", raw)
        if t and t not in anchors:
            anchors.append(t)

    return anchors
