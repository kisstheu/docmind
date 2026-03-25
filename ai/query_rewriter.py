from __future__ import annotations

import re
import requests


def _quick_rule_rewrite(question: str) -> str | None:
    q = (question or "").strip()
    q_norm = q.replace(" ", "")

    mismatch_patterns = ["不符", "不一致", "不匹配", "对不上", "冲突", "矛盾", "是否一致", "有没有一致"]
    if any(k in q_norm for k in mismatch_patterns):
        if "文件名" in q_norm or "标题" in q_norm:
            return "文件名 内容 不一致"
        if "正文" in q_norm:
            return "文件名 正文 不一致"
        if "名称" in q_norm:
            return "名称 内容 不一致"
        return "内容 文件名 不一致"

    # 稳定规则：最近有哪些和X有关的记录？
    related_record_patterns = [
        r"(?:最近)?有哪些?和(.+?)(?:有关|相关)(?:的)?(?:记录|文档|文件)",
        r"(.+?)(?:有关|相关)(?:的)?(?:记录|文档|文件)有哪些?",
    ]
    noise_words = {"最近", "最近的", "这些", "那些", "这个", "那个", "内容", "东西", "情况"}
    for pattern in related_record_patterns:
        m = re.search(pattern, q_norm)
        if not m:
            continue

        topic = (m.group(1) or "").strip("，。！？；：,.!?;:")
        topic = re.sub(r"^(和|与|跟|关于)", "", topic)
        if not topic:
            continue
        if topic in noise_words:
            continue
        if len(topic) > 18:
            topic = topic[:18]

        # 固定补上“记录/纪要/笔记”，减少改写随机性导致的漏召回。
        return f"{topic} 记录 纪要 笔记"

    return None


def _sanitize_search_query(search_query: str) -> str:
    q = (search_query or "").strip()

    q = re.sub(r'\.(txt|md|pdf|docx)$', '', q, flags=re.IGNORECASE).strip()
    q = re.sub(r'[^\w\s\u4e00-\u9fa5]', ' ', q)
    q = re.sub(r'\s+', ' ', q).strip()

    return q


def _is_too_generic(search_query: str) -> bool:
    q = (search_query or "").strip()
    if not q:
        return True

    tokens = [x for x in q.split() if x.strip()]
    if len(tokens) < 2:
        return True

    generic_tokens = {"内容", "文件名", "标题", "正文", "文档", "资料", "信息", "情况", "问题"}
    if all(t in generic_tokens for t in tokens):
        return True

    return False


def rewrite_search_query(question: str, memory_buffer, ollama_api_url: str, ollama_model: str, logger):
    quick = _quick_rule_rewrite(question)
    if quick:
        logger.debug(f"🔍 [本地规则重写]：{quick}")
        return quick

    history_str = "\n".join(memory_buffer[-4:])
    if not memory_buffer:
        task_desc = f"当前没有历史对话。请直接提取用户问题‘{question}’中的核心词作为搜索短语。绝对禁止脑补任何不存在的人名（如李四、张三）或项目名（如项目A）！"
    else:
        task_desc = (
            f"请结合上述历史，理解用户最新问题‘{question}’的真实搜索意图。\n"
            f"🚨【绝对禁止】：严禁直接复制或罗列历史对话中的实体列表（如一长串公司名、人名）！\n"
            f"💡【正确做法】：将其抽象为3-5个高度概括的搜索关键词"
        )

    rewrite_prompt = (
        f"【近期对话历史】\n{history_str}\n\n"
        f"【任务】\n{task_desc}\n"
        f"【最高警告】：\n"
        f"1. 绝对不允许脑补外部文件名！\n"
        f"2. 💡【转移话题判定】：如果用户暗示‘其他’、‘另外的’或‘从xxx出发’，必须立刻抛弃历史记录中的旧实体和旧文件名！\n"
        f"3. 🚨【实体保护原则】：如果用户最新提问中出现了明确的具体人名、地名或实体，重写后的搜索词【必须】包含该新实体，绝对不允许用历史记录中的旧名字去覆盖！\n"
        f"4. 🛑【禁止过度翻译】：如果用户输入了极短的英文字母，请【原封不动】地保留这些字母！绝对不允许脑补或翻译成词汇！也要防止将这些字母理解为文件后缀！\n"
        f"5. 🛑【输出格式绝对指令】：只能返回纯粹的词汇短语，词与词之间用空格隔开。绝对不允许输出完整的句子！绝对不允许包含问号、逗号、感叹号等任何标点符号！\n"
        f"6. 🕵️【侦探直觉与事实检索】：\n"
        f"   - 当用户询问‘人际关系’或‘都有谁’时，翻译为查找具体人名和事件。\n"
        f"7. 🚫【致命禁忌】：提取的搜索短语中【绝对不可以】包含“.txt”、“.md”、“.pdf”或“.docx”等扩展名！\n"
        f"8. 🗣️【上下文继承与纠错领悟】：如果用户输入极短（如‘A其实是B’），说明是在纠正上一轮的映射。你【必须】像人类一样，把上一轮的核心问题带入新的搜索词中！绝不能把核心意图弄丢！\n"
        f"9. 🤡【调侃与情绪过滤】：如果用户的最新提问带有表情包（如😄、😅、😂）或明显是随口调侃，请【仅保留】上一轮的核心实体（如人名）作为搜索词，绝对禁止脑补出“冲突”、“争执”、“暴力”等严肃词汇去污染搜索池！\n"
        f"10. 🏢【职场语义翻译】：如果用户询问某人的‘作品’、‘产出’、‘成果’或‘做过什么’，请务必将其翻译为具体的职场实体词汇，如：方案、文档、代码、系统、项目、设计。绝对不要只保留‘作品’这种偏文艺的词汇，避免导致技术文档检索失败！"
    )
    try:
        payload = {"model": ollama_model, "prompt": rewrite_prompt, "stream": False}
        response = requests.post(ollama_api_url, json=payload, timeout=10)
        response.raise_for_status()
        search_query = response.json().get("response", "").strip()
        search_query = _sanitize_search_query(search_query)

        if _is_too_generic(search_query):
            fallback = _quick_rule_rewrite(question)
            if fallback:
                logger.debug(f"🔁 [泛化结果回退到规则模板]：{fallback}")
                return fallback

        logger.debug(f"🔍 [本地引擎意图重写]：{search_query}")
        return search_query
    except Exception as e:
        fallback = _quick_rule_rewrite(question)
        if fallback:
            logger.warning(f"⚠️ 本地意图提取失败，启用规则兜底：{e}")
            return fallback

        logger.warning(f"⚠️ 本地意图提取失败，退回原问题：{e}")
        return question
