from __future__ import annotations

import re

# Keywords via unicode escapes to avoid source encoding issues.
_KW_FILE = (
    "\u6587\u4ef6",  # 文件
    "\u6587\u6863",  # 文档
    "doc",
    "docx",
    "txt",
    "pdf",
)
_KW_STRONG = (
    "\u53d8\u66f4\u8bb0\u5f55",  # 变更记录
    "\u53d8\u66f4\u5386\u53f2",  # 变更历史
    "\u64cd\u4f5c\u8bb0\u5f55",  # 操作记录
    "\u4fee\u6539\u8bb0\u5f55",  # 修改记录
    "\u5220\u9664\u8bb0\u5f55",  # 删除记录
    "\u91cd\u547d\u540d\u8bb0\u5f55",  # 重命名记录
    "changelog",
    "changehistory",
    "change log",
)
_KW_WEAK_CHANGE = (
    "\u53d8\u66f4",  # 变更
    "\u6539\u52a8",  # 改动
    "\u4fee\u6539",  # 修改
    "\u5220\u9664",  # 删除
    "\u91cd\u547d\u540d",  # 重命名
    "\u6539\u540d",  # 改名
)
_KW_META = (
    "\u8bb0\u5f55",  # 记录
    "\u5386\u53f2",  # 历史
    "\u6700\u8fd1",  # 最近
    "\u73b0\u5728",  # 现在
    "\u54ea\u4e9b",  # 哪些
    "\u67e5\u770b",  # 查看
    "\u5217\u51fa",  # 列出
)


def _normalize_text(text: str) -> str:
    q = (text or "").strip().lower()
    if not q:
        return ""
    q = q.replace(" ", "")
    return q


def is_change_history_query(question: str) -> bool:
    q = _normalize_text(question)
    if not q:
        return False

    has_file_ctx = any(k in q for k in _KW_FILE)
    if any(k in q for k in _KW_STRONG):
        return True if has_file_ctx else ("changelog" in q or "changehistory" in q)

    if not has_file_ctx:
        return False

    if any(k in q for k in _KW_WEAK_CHANGE) and any(k in q for k in _KW_META):
        return True

    # e.g. 现在有哪些文件发生了变更
    if re.search(r"(?:\u54ea\u4e9b|\u6700\u8fd1|\u73b0\u5728).*(?:\u6587\u4ef6|\u6587\u6863).*(?:\u53d8\u66f4|\u6539\u52a8|\u4fee\u6539)", q):
        return True

    return False


def extract_history_limit(question: str, default: int = 20, max_limit: int = 100) -> int:
    q = _normalize_text(question)
    if not q:
        return default

    if "\u5168\u90e8" in q:  # 全部
        return max_limit

    m = re.search(r"(\d+)\s*(?:\u6761|\u4e2a|\u4efd)?", q)
    if not m:
        return default

    n = int(m.group(1))
    return max(1, min(n, max_limit))
