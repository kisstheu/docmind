from __future__ import annotations

import re


ASSISTANT_PRODUCT_NAME = "DocMind"
ASSISTANT_IDENTITY_REPLY = (
    "我是 DocMind，一款面向本地资料检索、整理与有限归纳的 AI 助手。"
)

_IDENTITY_PROVENANCE_MARKERS = (
    "模型",
    "model",
    "开发",
    "研发",
    "哪个公司",
    "哪家公司",
    "厂商",
    "供应商",
    "provider",
)
_IDENTITY_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^(?:请问|想问一下|我想问一下)?(?:你|您)(?:到底|究竟)?(?:是)?(?:谁|哪位)(?:啊|呀|呢)?$",
        r"^(?:请问|想问一下|我想问一下)?(?:你|您)(?:到底|究竟)?是(?:什么|啥)(?:东西)?(?:啊|呀|呢)?$",
        r"^(?:请问|想问一下|我想问一下)?(?:你|您)(?:叫(?:什么|啥)(?:名字|名)?|(?:的)?(?:名字|名称|姓名)(?:是|叫)?(?:什么|啥)?)(?:啊|呀|呢)?$",
        r"^(?:请问|想问一下|我想问一下)?(?:你|您)(?:到底|究竟)?是(?:一个|个)?(?:什么|啥|哪种|哪个)(?:样的)?(?:ai)?助手(?:啊|呀|呢)?$",
        r"^(?:请|请你|麻烦你)?(?:简单)?(?:自我介绍|介绍一下你自己|介绍下你自己)(?:一下|下)?(?:吧|啊|呀)?$",
        r"^(?:whoareyou|whatisyourname|whatsyourname|whatkindofassistantareyou|introduceyourself)$",
    )
)


def _normalize_identity_question(question: str) -> str:
    normalized = (question or "").strip().lower()
    normalized = re.sub(r"[？?！!，,。\.、：:；;\s'’]+", "", normalized)
    return normalized


def is_assistant_identity_request(question: str) -> bool:
    """Return whether the user is asking for DocMind's product identity."""
    q = _normalize_identity_question(question)
    if not q or any(marker in q for marker in _IDENTITY_PROVENANCE_MARKERS):
        return False
    return any(pattern.fullmatch(q) for pattern in _IDENTITY_PATTERNS)


def answer_assistant_identity_question(question: str) -> str | None:
    """Answer product identity locally so a provider persona cannot replace it."""
    if not is_assistant_identity_request(question):
        return None
    return ASSISTANT_IDENTITY_REPLY


__all__ = [
    "ASSISTANT_IDENTITY_REPLY",
    "ASSISTANT_PRODUCT_NAME",
    "answer_assistant_identity_question",
    "is_assistant_identity_request",
]
