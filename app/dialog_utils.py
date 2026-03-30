from __future__ import annotations

import re


FOLLOWUP_PATTERNS = [
    "这个", "那个", "这些", "那些",
    "哪些", "哪类", "哪几类", "哪方面", "哪些方面",
    "具体", "详细", "展开", "细说", "继续",
    "还有", "然后", "再说", "再展开",
    "怎么分", "如何分", "分成什么", "分成哪些",

    "太细了", "太碎了", "别那么细", "不要太细",
    "粗一点", "粗一些", "再粗一点", "再粗一些",
    "大一点", "大一些", "再大一点", "再大一些",
    "概括一下", "再概括一下",

    "列一下", "列一下吧", "列出来", "展开一下", "展开列一下",
]

ACK_PATTERNS = [
    "不少", "挺多", "很多", "可以", "不错", "厉害", "牛", "行", "好", "好的",
]

SMALLTALK_PATTERNS = [
    "你好", "嗨", "hello", "hi", "在吗", "在不在",
    "谢谢", "感谢", "多谢", "辛苦了",
    "哈哈", "呵呵", "hhh", "hh",
    "好的", "行", "ok",
    "厉害", "真强", "挺强", "牛", "不错", "可以啊", "这么强", "赞", "棒",
    "太强", "你太强", "你可太强", "太厉害",
    "高冷", "好高冷", "太高冷", "冷淡", "太冷淡",
]

SMALLTALK_STRONG_PATTERNS = [
    "谢谢", "感谢", "多谢", "辛苦了",
    "你好", "在吗",
    "厉害", "真强", "挺强", "这么强", "太强", "太厉害",
]

SMALLTALK_BLOCK_PATTERNS = [
    "找", "查", "搜", "检索",
    "文件", "文档", "资料", "记录",
    "公司", "人物", "人名", "项目",
    "时间", "日期", "最近", "最早", "最晚",
    "多少", "哪些", "哪几个", "哪几家",
    "列出", "清单", "提到", "提及", "在哪", "位置",
    "帮我", "麻烦", "请你",
]

EMOJI_MARKERS = "😀😁😂🤣😃😄😅😊🙂😉😍😘😎👍👏🙏❤❤️"


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[？?！!，,。\.、\s]+", "", text)
    return text


def is_followup_question(question: str) -> bool:
    q = normalize_text(question)

    if not q:
        return False

    # 太长的一般不是“弱追问”
    if len(q) > 14:
        return False

    if any(p in q for p in FOLLOWUP_PATTERNS):
        return True

    return False
def is_repo_meta_confirmation(question: str) -> bool:
    """
    判断是否属于“基于上一轮知识库分类结果做确认/转述”的追问
    例如：
    - 也就是说关于劳动关系的比较多咯？
    - 所以技术类也不少？
    - 那就是说工作相关占多数？
    """
    q = (question or "").strip()
    if not q:
        return False

    patterns = [
        "也就是说", "也就是", "所以说", "所以", "那就是说", "那就是",
        "这么看", "看来", "按你这么说",
        "比较多", "更多", "占多数", "占大头", "为主", "主要是",
        "是不是", "对吗", "咯", "吗"
    ]

    # 短句 + 带确认/归纳口吻
    return len(q) <= 40 and any(p in q for p in patterns)


def is_judgment_request(question: str) -> bool:
    """
    判断是否属于“基于已有材料请求评价/判断”
    例如：
    - 是他胡闹吗
    - 这合理吗
    - 谁更有问题
    - 我能赢吗
    """
    q = (question or "").strip()
    if not q:
        return False

    patterns = [
        "胡闹吗", "是不是胡闹", "是他胡闹吗", "是她胡闹吗",
        "离谱吗", "合理吗", "过分吗",
        "谁的问题", "谁有问题", "谁更过分", "谁更离谱",
        "算不算", "是不是在甩锅", "是不是他的错", "是不是她的错",
        "是不是公司的问题", "是不是他的问题", "是不是她的问题",
        "你觉得谁", "你研究一下", "你判断一下", "你怎么看",

        "我能赢吗", "能赢吗", "能不能赢",
        "有胜算吗", "胜算大吗", "赢面大吗", "把握大吗",
        "会输吗", "结果会怎样", "结果怎么样",
    ]
    return any(p in q for p in patterns)

def is_acknowledgement(question: str) -> bool:
    q = normalize_text(question)
    if len(q) > 10:
        return False
    return any(p in q for p in ACK_PATTERNS)


def is_smalltalk_message(question: str) -> bool:
    q = normalize_text(question)
    if not q:
        return False

    assistant_persona_markers = (
        "你多大",
        "你几岁",
        "你多少岁",
        "你今年多大",
        "你的年龄",
        "你的年纪",
        "你叫什么",
        "你叫啥",
        "你叫啥名",
        "你叫啥名字",
        "你的名字",
        "你的姓名",
        "whatisyourname",
        "yourname",
        "howoldareyou",
        "yourage",
    )
    if any(p in q for p in assistant_persona_markers):
        return True

    has_smalltalk_pattern = any(p in q for p in SMALLTALK_PATTERNS)
    has_emoji_marker = any(ch in q for ch in EMOJI_MARKERS)
    if not has_smalltalk_pattern and not has_emoji_marker:
        return False

    if any(p in q for p in SMALLTALK_BLOCK_PATTERNS):
        return False

    if len(q) <= 24:
        return True

    return any(p in q for p in SMALLTALK_STRONG_PATTERNS)

def is_action_request(question: str) -> bool:
    q = (question or "").strip()

    patterns = [
        "研究", "研究下", "研究一下",
        "分析", "分析下", "分析一下",
        "找下", "找一下", "找找", "找出", "找出来",
        "看看", "看下", "看一下", "帮我看",
        "查下", "查一下", "查一查",
        "查找", "查找下", "查找一下",
        "总结下", "总结一下",
        "整理下", "整理一下",
        "说说", "讲讲", "解释一下",
        "聊下", "聊一下", "聊聊",
        "梳理下", "梳理一下",
    ]

    return any(p in q for p in patterns)



def is_content_followup_question(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False

    patterns = [
        "然后呢", "后来呢", "接着呢", "结果呢",
        "他呢", "她呢", "公司呢", "对方呢",
        "他做了什么", "她做了什么", "公司做了什么",
        "做了哪些处理", "做了什么处理", "怎么处理的",
        "具体呢", "具体说说", "展开说说",
        "为什么", "依据是什么", "哪些地方", "有哪些证据",
        "这边呢", "那边呢", "后来怎么说", "后来怎么处理",
    ]

    if any(p in q for p in patterns):
        return True

    # 很短 + 带明显承接意味
    if len(q) <= 20 and any(x in q for x in ["公司", "他", "她", "对方", "后来", "处理", "证据", "依据"]):
        return True

    return False

def is_relationship_analysis_request(question: str) -> bool:
    q = (question or "").strip()

    patterns = [
        "关系相对好", "关系比较好", "关系好吗",
        "对我比较好", "对我好一些", "态度好一些",
        "谁比较支持我", "谁更站我这边", "谁态度更好",
        "谁更愿意沟通", "谁没那么强硬", "谁对我还行",
    ]

    return any(p in q for p in patterns)


def is_query_correction(question: str) -> bool:
    q = (question or "").strip()

    patterns = [
        "我是说", "我说的是", "不是这个",
        "不是这个意思", "不是说这个",
        "你理解错了", "你搞错了",
        "我指的是", "我问的是",
        "换句话说", "我的意思是",
    ]

    return any(p in q for p in patterns)


def is_structured_output_request(question: str) -> bool:
    q = (question or "").strip()
    patterns = [
        "时间线", "按时间顺序", "梳理一下", "整理一下",
        "列个清单", "分点总结", "做个表", "列出来",
        "给我个脉络", "帮我归纳一下",
    ]
    return any(p in q for p in patterns)
