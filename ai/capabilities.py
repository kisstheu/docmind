from __future__ import annotations

from collections import Counter
from pathlib import Path


def _normalize_tag(tag: str) -> str:
    tag = (tag or "").strip().lower()
    if not tag:
        return ""

    # 去掉一些明显无意义的标签
    bad_tags = {
        "项目", "文件", "文档", "内容", "记录", "笔记",
        "开发", "系统", "设计", "方案", "总结",
        "文本", "资料", "信息", "问题", "相关"
    }
    if tag in bad_tags:
        return ""

    if len(tag) < 2:
        return ""

    return tag


def _merge_similar_tags(tag_counter: Counter) -> Counter:
    merged = Counter()

    for tag, count in tag_counter.most_common():
        placed = False

        for existing in list(merged.keys()):
            # 完全相同
            if tag == existing:
                merged[existing] += count
                placed = True
                break

            # 包含关系：短词是长词核心
            if len(tag) >= 2 and len(existing) >= 2:
                if tag in existing or existing in tag:
                    # 优先保留更长、更具体的那个
                    better = existing if len(existing) >= len(tag) else tag
                    worse = tag if better == existing else existing

                    merged[better] += merged[worse] + count
                    if worse in merged:
                        del merged[worse]

                    placed = True
                    break

        if not placed:
            merged[tag] += count

    return merged


def _coarse_category_from_path(path_str: str) -> str:
    p = Path(path_str)

    if len(p.parts) >= 2:
        # 优先按顶层目录
        return p.parts[0]

    suffix = p.suffix.lower()
    if suffix:
        return suffix

    return "其他"


def _is_category_summary_request(q: str) -> bool:
    q = (q or "").strip().lower()
    return any(x in q for x in [
        "大方面", "大类",
        "粗一点", "粗一些", "再粗一点", "再粗一些",
        "大一点", "大一些", "再大一点", "再大一些",
        "概括", "概括一下", "再概括一下",
        "太碎", "太细", "太细了", "别那么细", "不要太细",
    ])


def _is_category_confirmation_request(q: str) -> bool:
    q = (q or "").strip().lower()
    return any(x in q for x in [
        "也就是说", "也就是", "所以", "所以说", "那就是说", "那就是",
        "这么看", "看来", "按你这么说",
        "比较多", "更多", "占多数", "占大头", "主要是", "为主",
        "是不是", "对吗",
    ])



def _extract_fine_topics(repo_state):
    doc_records = getattr(repo_state, "doc_records", None) or []
    print(f"📦 doc_records 数量: {len(doc_records)}")
    if doc_records:
        print(f"📦 示例: {doc_records[0]}")

    if not doc_records:
        return []

    raw_counter = Counter()

    for record in doc_records:
        shadow_tags = record.get("shadow_tags", "") or ""
        tags = [x.strip() for x in shadow_tags.split() if x.strip()]

        # 一个文件内部去重，避免同一文件重复加权太多
        unique_tags = set()

        for tag in tags:
            norm = _normalize_tag(tag)
            if norm:
                unique_tags.add(norm)

        for norm in unique_tags:
            raw_counter[norm] += 1

    if not raw_counter:
        return []

    merged_counter = _merge_similar_tags(raw_counter)
    return merged_counter.most_common()


def _validate_summary_output(result: str) -> list[str]:
    text = (result or "").strip()
    if not text:
        raise RuntimeError("本地模型返回为空")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_lines = [line for line in lines if line.startswith("- ")]

    if len(bullet_lines) < 2:
        raise RuntimeError(f"本地模型输出不符合预期: {text}")

    return bullet_lines[:6]


def summarize_topics_coarsely_with_local_llm(fine_topics, topic_summarizer):
    """
    fine_topics: List[Tuple[str, int]]
    topic_summarizer: callable(prompt:str) -> str
    """
    if not topic_summarizer:
        raise RuntimeError("未提供 topic_summarizer，本地模型概括不可用")

    top_topics = [name for name, _ in fine_topics[:12]]
    if not top_topics:
        raise RuntimeError("没有可用于概括的细主题")

    prompt = (
        "下面是个人知识库中出现频率较高的一批细主题。\n"
        "请把它们压缩概括成 4 到 6 个更大的内容方面。\n"
        "必须遵守：\n"
        "1. 只输出列表\n"
        "2. 每行必须以“- ”开头\n"
        "3. 不要写前言，不要解释，不要总结\n"
        "4. 不要逐条复述原始细主题\n"
        "5. 要把相近内容合并成更大的方面\n"
        "6. 用自然中文，不要太学术\n"
        "7. 表达像人在整理自己的文档，而不是写报告\n\n"
        "细主题：\n"
        + "\n".join(f"- {x}" for x in top_topics)
    )

    result = topic_summarizer(prompt)
    bullet_lines = _validate_summary_output(result)

    return (
        "按更大的方面看，当前知识库主要集中在这些板块：\n"
        + "\n".join(bullet_lines)
    )




def classify_repo_meta_question(question: str, last_user_question: str | None = None) -> str | None:
    q = question.strip().lower()

    if any(x in q for x in ["多少文件", "多少个文件", "文件数量", "文档数量"]):
        return "count"

    if any(x in q for x in ["哪些格式", "文件格式", "文档格式", "支持格式"]):
        return "format"

    if any(x in q for x in ["最近更新", "最新文件", "最早文件"]):
        return "time"

    if any(x in q for x in ["有哪些文档", "都有哪些文档", "文档有哪些", "有哪些文件", "都有哪些文件", "文件有哪些", "文档清单", "文件清单", "列出文档", "列出文件"]):
        return "list_files"

    # 先判定“更粗的大方面”
    if _is_category_summary_request(q):
        return "category_summary"

    if _is_category_confirmation_request(q):
        return "category_confirm"

    if any(x in q for x in ["哪些方面", "哪几类", "哪些类别", "怎么分类", "如何分类", "分成什么", "分成哪些", "按什么分", "大致分", "可以分成"]):
        return "category"

    if last_user_question:
        last_q = last_user_question.strip().lower()

        # 从列文件追问到分类
        if any(x in last_q for x in ["有哪些文档", "有哪些文件", "文档清单", "文件清单", "列出文档", "列出文件"]):
            if any(x in q for x in ["方面", "分类", "类别", "哪类", "怎么分", "如何分"]):
                return "category"

        # 从分类继续追问到更大的方面 / 做结论确认
        if any(x in last_q for x in [
            "哪些方面", "哪几类", "哪些类别", "怎么分类", "如何分类",
            "分成什么", "分成哪些", "按什么分", "大致分", "可以分成",
            "大方面", "大类", "粗一点", "概括", "太碎", "别那么细", "不要太细", "再大一点"
        ]):
            if _is_category_summary_request(q):
                return "category_summary"
            if _is_category_confirmation_request(q):
                return "category_confirm"

    return None

def answer_repo_content_category_question(repo_state):
    fine_topics = _extract_fine_topics(repo_state)
    if not fine_topics:
        return "当前标签信息不足，暂时还无法按内容主题分类。"

    top_k = 12
    lines = [f"- {tag}：约 {count} 个文件" for tag, count in fine_topics[:top_k]]

    return (
        "按内容主题粗略来看，当前知识库主要集中在这些方面：\n"
        + "\n".join(lines)
        + "\n\n这是基于影子标签自动归纳出来的，不是按文件后缀硬分的。"
    )

def answer_repo_content_category_summary_question(repo_state, topic_summarizer):
    fine_topics = _extract_fine_topics(repo_state)
    if not fine_topics:
        return "当前标签信息不足，暂时还无法按更大的方面概括。"

    return summarize_topics_coarsely_with_local_llm(
        fine_topics=fine_topics,
        topic_summarizer=topic_summarizer,
    )


def _extract_confirmation_candidates(question: str) -> list[str]:
    q = (question or "").strip().lower()
    if not q:
        return []

    stop_phrases = [
        "也就是说", "也就是", "所以", "所以说", "那就是说", "那就是",
        "这么看", "看来", "按你这么说",
        "比较多", "更多", "占多数", "占大头", "主要是", "为主",
        "是不是", "对吗", "咯", "吗", "呢",
        "关于", "相关", "内容", "这一类", "这类",
    ]

    for s in stop_phrases:
        q = q.replace(s, " ")

    raw_parts = [x.strip() for x in q.replace("，", " ").replace("。", " ").replace("？", " ").split() if x.strip()]

    # 再按常见连接词拆一下
    parts = []
    for p in raw_parts:
        for seg in p.replace("的", " ").replace("了", " ").split():
            seg = seg.strip()
            if seg:
                parts.append(seg)

    # 去掉太短和无意义词
    bad = {
        "这个", "那个", "这些", "那些",
        "东西", "方面", "类别", "分类", "情况",
        "比较", "就是", "还是", "应该", "感觉",
    }

    candidates = []
    for p in parts:
        if len(p) < 2:
            continue
        if p in bad:
            continue
        candidates.append(p)

    # 去重保序
    seen = set()
    result = []
    for x in candidates:
        if x not in seen:
            seen.add(x)
            result.append(x)

    return result

def _expand_candidate_fragments(cand: str) -> list[str]:
    cand = (cand or "").strip()
    if not cand:
        return []

    fragments = {cand}

    # 常见中文后缀去壳
    suffixes = ["方面", "相关", "问题", "情况", "内容", "这块", "这一块"]
    for s in suffixes:
        if cand.endswith(s) and len(cand) > len(s) + 1:
            fragments.add(cand[:-len(s)])

    # 对 4~6 字中文短语，补充 2~4 字片段
    n = len(cand)
    if 4 <= n <= 6:
        for size in range(2, min(4, n) + 1):
            for i in range(0, n - size + 1):
                frag = cand[i:i + size]
                if len(frag) >= 2:
                    fragments.add(frag)

    # 去掉太泛的碎片
    bad = {"关系", "问题", "情况", "内容", "方面", "相关", "最近"}
    result = [x for x in fragments if x not in bad and len(x) >= 2]

    # 长的优先，避免先命中太短碎片
    result.sort(key=len, reverse=True)
    return result

def _match_confirmation_candidates_to_topics(candidates: list[str], fine_topics: list[tuple[str, int]]):
    matches = []

    for cand in candidates:
        fragments = _expand_candidate_fragments(cand)
        best_match = None

        for topic, count in fine_topics:
            for frag in fragments:
                if frag in topic or topic in frag:
                    if best_match is None or count > best_match[2]:
                        best_match = (cand, topic, count)
                    break

        if best_match:
            matches.append(best_match)

    return matches

def answer_repo_content_category_confirm_question(question: str, repo_state) -> str:
    fine_topics = _extract_fine_topics(repo_state)
    if not fine_topics:
        return "当前标签信息不足，我还没法判断这个方面是不是明显更多。"

    candidates = _extract_confirmation_candidates(question)
    if not candidates:
        top_preview = "、".join(name for name, _ in fine_topics[:3])
        return f"我大概明白你的意思，不过这句话里可直接比对的主题词不够明显。当前更突出的主题大致有：{top_preview}。"

    matches = _match_confirmation_candidates_to_topics(candidates, fine_topics)

    if not matches:
        top_preview = "、".join(f"{name}（约{count}）" for name, count in fine_topics[:5])
        return (
            f"不一定。按当前影子标签归纳的结果，"
            f"我还看不出“{'、'.join(candidates)}”是最突出的那一类。"
            f"目前更靠前的主题有：{top_preview}。"
        )

    # 取命中里最靠前/计数最高的一个
    best_cand, best_topic, best_count = sorted(matches, key=lambda x: x[2], reverse=True)[0]
    top_count = fine_topics[0][1] if fine_topics else best_count

    if best_count >= max(3, int(top_count * 0.7)):
        return (
            f"可以这么理解。按当前影子标签归纳结果看，"
            f"“{best_topic}”这一类确实算比较突出的主题之一，"
            f"大约对应 {best_count} 个文件。"
        )

    return (
        f"可以稍微这样理解，但没到特别压倒性的程度。"
        f"按当前结果看，和你这句话最接近的是“{best_topic}”，"
        f"大约对应 {best_count} 个文件。"
    )


def answer_repo_meta_question(
    question: str,
    repo_state,
    last_user_question: str | None = None,
    topic_summarizer=None,
):
    q = question.strip().lower()

    paths = list(repo_state.paths)
    all_files = list(repo_state.all_files)
    file_times = list(repo_state.file_times)

    if not paths:
        return "当前知识库里还没有可用文档。", "empty"

    topic = classify_repo_meta_question(question, last_user_question=last_user_question)

    if topic == "count":
        return f"当前知识库里共有 {len(paths)} 个可用文件。", topic

    if topic == "format":
        suffixes = sorted({f.suffix.lower() or '[无后缀]' for f in all_files})
        answer = "当前知识库中的文件格式有：\n" + "\n".join(f"- {s}" for s in suffixes)
        return answer, topic

    if topic == "time":
        latest_idx = max(range(len(file_times)), key=lambda i: file_times[i])
        earliest_idx = min(range(len(file_times)), key=lambda i: file_times[i])

        answer = (
            f"最早的文件：{paths[earliest_idx]}（{file_times[earliest_idx].strftime('%Y-%m-%d %H:%M:%S')}）\n"
            f"最新的文件：{paths[latest_idx]}（{file_times[latest_idx].strftime('%Y-%m-%d %H:%M:%S')}）"
        )
        return answer, topic

    if topic == "list_files":
        show_n = min(50, len(paths))
        preview = "\n".join(f"- {p}" for p in paths[:show_n])

        if len(paths) > show_n:
            answer = f"当前知识库里共有 {len(paths)} 个文件，先列出前 {show_n} 个：\n{preview}\n- ...(其余 {len(paths) - show_n} 个未展开)"
        else:
            answer = f"当前知识库里的文件如下：\n{preview}"

        return answer, topic

    if topic == "category":
        answer = answer_repo_content_category_question(repo_state)
        return answer, topic

    if topic == "category_summary":
        answer = answer_repo_content_category_summary_question(
            repo_state,
            topic_summarizer=topic_summarizer,
        )
        return answer, topic

    if topic == "category_confirm":
        answer = answer_repo_content_category_confirm_question(question, repo_state)
        return answer, topic

    return None, None


def answer_system_capability_question(question: str):
    q = (question or "").strip().lower()

    if any(k in q for k in ["你是谁", "介绍一下"]):
        return "我是你的 DocMind 随身助理，主要负责根据你的本地笔记和文档回答问题、整理线索，并做有限归纳。"

    if any(k in q for k in [
        "能干啥", "你能做什么", "能做什么", "可以做什么", "你可以做什么",
        "你的功能", "有什么功能", "有啥功能", "怎么用"
    ]):
        return (
            "我现在主要能做这几类事：\n"
            "1. 根据你的本地笔记、文档和资料回答具体问题。\n"
            "2. 帮你查某个人、某个项目、某段经历、某份材料里提到过什么。\n"
            "3. 帮你做有限总结，比如梳理某个方案、某组记录或某段时间线。\n"
            "4. 帮你做一些仓库层面的统计，比如文件数量、文件格式、最近更新情况等。\n"
            "5. 对于明显是闲聊、寒暄或系统介绍类问题，我也可以直接回答，不必走文档检索。"
        )

    return None

def answer_smalltalk(question: str, dialog_state=None) -> str | None:
    q = (question or "").strip().lower()

    judgment_like = [
        "胡闹吗", "合理吗", "过分吗", "离谱吗",
        "谁的问题", "谁有问题", "算不算",
        "是不是", "对吗", "你怎么看", "你研究一下", "你判断一下",
    ]
    if any(x in q for x in judgment_like):
        return None

    if any(x in q for x in ["你好", "嗨", "hello", "hi"]):
        return "你好呀。"

    if any(x in q for x in ["在吗", "在不在", "在嘛"]):
        return "在。"

    if any(x in q for x in ["谢谢", "多谢", "感谢", "辛苦了"]):
        return "不客气。"

    if any(x in q for x in ["不少", "挺多", "很多"]):
        if dialog_state and getattr(dialog_state, "last_route", None) == "repo_meta":
            if getattr(dialog_state, "last_local_topic", None) == "list_files":
                return "是的，现在已经积累得挺多了。"
        return "嗯，确实不少。"

    if any(x in q for x in ["太细了", "太碎了", "别那么细", "不要太细"]):
        if dialog_state and getattr(dialog_state, "last_route", None) == "repo_meta":
            if getattr(dialog_state, "last_local_topic", None) in {"category", "category_summary"}:
                return "嗯，我按更大的方面再给你归并一下。"
        return "嗯，我换个更粗的角度说。"

    if q in {"😁", "😄", "😆", "😂", "🤣", "😊", "😅", "🙂", "😉"}:
        if dialog_state and getattr(dialog_state, "last_route", None) == "system_capability":
            return "先别急着夸，真问起来我再表现。"
        return "哈哈。"

    if any(x in q for x in ["挺强", "真强", "厉害", "牛", "不错", "可以啊", "真行", "这么强", "看着挺强"]):
        if dialog_state and getattr(dialog_state, "last_route", None) == "system_capability":
            return "功能先说在前面，真干活还得看你怎么使唤我。"
        return "你这么一说，我都有点不好意思了。"

    if any(x in q for x in ["不谦虚", "挺自信", "还真敢讲", "你还挺会说"]):
        return "先把活干明白再谦虚，也不迟。"

    if any(x in q for x in ["好", "好的", "行", "可以", "明白了", "知道了"]):
        return "好。"

    if any(x in q for x in ["哈哈", "嘿嘿", "笑死", "有意思"]):
        return "哈哈。"

    return "嗯。"