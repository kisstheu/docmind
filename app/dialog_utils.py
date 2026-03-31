from __future__ import annotations

import re

from ai.capability_smalltalk import answer_smalltalk
from ai.query_rewriter import is_local_smalltalk_intent


FOLLOWUP_PATTERNS = [
    "这个",
    "那个",
    "这些",
    "那些",
    "哪些",
    "哪类",
    "具体",
    "详细",
    "展开",
    "细说",
    "继续",
    "还有",
    "然后",
    "再说",
    "再展开",
    "怎么分",
    "如何分",
    "分成什么",
    "分成哪些",
    "太细了",
    "别那么细",
    "不要太细",
    "粗一点",
    "再粗一点",
    "大一点",
    "再大一点",
    "概括一下",
    "再概括一下",
    "列一下",
    "列出来",
]


def normalize_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[，。！？、,.!?；;：:\s]+", "", text)
    return text


def is_followup_question(question: str) -> bool:
    q = normalize_text(question)
    if not q:
        return False
    if len(q) > 14:
        return False
    return any(p in q for p in FOLLOWUP_PATTERNS)


def is_repo_meta_confirmation(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False

    patterns = [
        "也就是说",
        "所以说",
        "所以",
        "那就是说",
        "这么看",
        "看来",
        "按你这么说",
        "比较多",
        "更多",
        "占多数",
        "占大头",
        "为主",
        "主要是",
        "是不是",
        "对吗",
        "吗",
    ]
    return len(q) <= 40 and any(p in q for p in patterns)


def is_judgment_request(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False

    patterns = [
        "胡闹吗",
        "是不是胡闹",
        "合理吗",
        "过分吗",
        "离谱吗",
        "谁的问题",
        "谁有问题",
        "谁更过分",
        "谁更离谱",
        "算不算",
        "是不是在甩锅",
        "是不是他的错",
        "是不是她的错",
        "是不是公司的问题",
        "你觉得谁",
        "你判断一下",
        "你怎么看",
        "我能赢吗",
        "能赢吗",
        "能不能赢",
        "有胜算吗",
        "胜算大吗",
        "把握大吗",
        "会输吗",
        "结果会怎样",
    ]
    return any(p in q for p in patterns)


def is_smalltalk_message(question: str) -> bool:
    q = normalize_text(question)
    if not q:
        return False

    if answer_smalltalk(question) is not None:
        return True

    if is_local_smalltalk_intent(question):
        return True

    fallback_smalltalk_terms = (
        "谢谢",
        "感谢",
        "晚安",
        "你好",
        "很强",
        "太强",
        "高冷",
        "哈哈",
        "hh",
        "hhh",
    )
    retrieval_terms = (
        "找",
        "查",
        "搜",
        "检索",
        "文件",
        "文档",
        "资料",
        "记录",
        "公司",
        "项目",
        "人物",
        "人名",
        "列出",
        "清单",
        "提到",
        "提及",
    )
    emoji_markers = "😀😁😂🤣😃😄😅😊🙂😉😍😘😎👍👏🙏❤❤️"

    if any(p in q for p in retrieval_terms):
        return False
    if len(q) <= 24 and any(p in q for p in fallback_smalltalk_terms):
        return True
    return len(q) <= 24 and any(ch in question for ch in emoji_markers)


def is_action_request(question: str) -> bool:
    q = (question or "").strip()
    patterns = [
        "研究",
        "分析",
        "找下",
        "找一下",
        "找",
        "看下",
        "看看",
        "查下",
        "查一下",
        "查找",
        "总结一下",
        "整理一下",
        "解释一下",
        "聊下",
        "梳理一下",
    ]
    return any(p in q for p in patterns)


def is_content_followup_question(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False

    patterns = [
        "然后呢",
        "后来呢",
        "接着呢",
        "结果呢",
        "他呢",
        "她呢",
        "公司呢",
        "对方呢",
        "他做了什么",
        "她做了什么",
        "公司做了什么",
        "具体呢",
        "展开说说",
        "为什么",
        "依据是什么",
        "有哪些证据",
        "后来怎么说",
        "后来怎么处理",
    ]
    if any(p in q for p in patterns):
        return True

    return len(q) <= 20 and any(x in q for x in ["公司", "他", "她", "对方", "后来", "处理", "证据", "依据"])


def is_relationship_analysis_request(question: str) -> bool:
    q = (question or "").strip()
    patterns = [
        "关系相对好",
        "关系比较好",
        "关系好吗",
        "对我比较好",
        "态度好一点",
        "谁比较支持我",
        "谁更站我这边",
        "谁态度更好",
        "谁更愿意沟通",
        "谁没那么强硬",
        "谁对我还行",
    ]
    return any(p in q for p in patterns)


def is_query_correction(question: str) -> bool:
    q = (question or "").strip()
    patterns = [
        "我是说",
        "我说的是",
        "不是这个",
        "不是这个意思",
        "不是说这个",
        "你理解错了",
        "你搞错了",
        "我指的是",
        "我问的是",
        "换句话说",
        "我的意思是",
    ]
    return any(p in q for p in patterns)


def is_structured_output_request(question: str) -> bool:
    q = (question or "").strip()
    patterns = [
        "时间线",
        "按时间顺序",
        "梳理一下",
        "整理一下",
        "列个清单",
        "分点总结",
        "做个表",
        "列出来",
        "给我个总结",
        "帮我归纳一下",
    ]
    return any(p in q for p in patterns)
