from __future__ import annotations

from ai.capability_common import SYSTEM_CAPABILITY_KEYWORDS, clean_text, contains_any


def answer_system_capability_question(question: str):
    q = clean_text(question)

    if contains_any(q, ("你是谁", "介绍一下")):
        return "我是你的 DocMind 随身助理，主要负责根据你的本地笔记和文档回答问题、整理线索，并做有限归纳。"

    if contains_any(q, SYSTEM_CAPABILITY_KEYWORDS):
        return (
            "我现在主要能做这几类事：\n"
            "1. 根据你的本地笔记、文档和资料回答具体问题。\n"
            "2. 帮你查某个人、某个项目、某段经历、某份材料里提到过什么。\n"
            "3. 帮你做有限总结，比如梳理某个方案、某组记录或某段时间线。\n"
            "4. 帮你做一些仓库层面的统计，比如文件数量、文件格式、最近更新情况等。\n"
            "5. 对于明显是闲聊、寒暄或系统介绍类问题，我也可以直接回答，不必走文档检索。"
        )

    return None
