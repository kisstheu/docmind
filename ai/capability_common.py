from __future__ import annotations

from collections import Counter
from typing import Callable, Iterable, Sequence

BAD_TAGS = {
    "项目", "文件", "文档", "内容", "记录", "笔记",
    "开发", "系统", "设计", "方案", "总结",
    "文本", "资料", "信息", "问题", "相关",
}

CATEGORY_SUMMARY_KEYWORDS = (
    "大方面", "大类",
    "粗一点", "粗一些", "再粗一点", "再粗一些",
    "大一点", "大一些", "再大一点", "再大一些",
    "概括", "概括一下", "再概括一下",
    "太碎", "太细", "太细了", "别那么细", "不要太细",
)

CATEGORY_CONFIRM_KEYWORDS = (
    "也就是说", "也就是", "所以", "所以说", "那就是说", "那就是",
    "这么看", "看来", "按你这么说",
    "比较多", "更多", "占多数", "占大头", "主要是", "为主",
    "是不是", "对吗",
)

CATEGORY_KEYWORDS = (
    "哪些方面", "哪几类", "哪些类别", "怎么分类", "如何分类",
    "分成什么", "分成哪些", "按什么分", "大致分", "可以分成",

    "有哪些类文档", "哪些类文档", "文档有哪些类",
    "有哪些类文件", "哪些类文件", "文件有哪些类",
    "文档分哪几类", "文件分哪几类",
    "有哪些类别的文档", "有哪些类别的文件",
)

LIST_FILE_KEYWORDS = (
    "有哪些文档", "都有哪些文档", "文档有哪些",
    "有哪些文件", "都有哪些文件", "文件有哪些",
    "文档清单", "文件清单", "列出文档", "列出文件",

    "列一下", "列一下吧", "列出来", "展开一下", "展开列一下",
)

TOTAL_SIZE_KEYWORDS = (
    "总共占多大空间", "一共占多大空间", "占多大空间",
    "总大小", "一共多大", "总共多大",
    "总体积", "总共多少体积",
    "占用空间", "总占用", "总容量",
    "文档多大", "文件多大", "文档总大小", "文件总大小",
)

CONFIRMATION_STOP_PHRASES = (
    "也就是说", "也就是", "所以", "所以说", "那就是说", "那就是",
    "这么看", "看来", "按你这么说",
    "比较多", "更多", "占多数", "占大头", "主要是", "为主",
    "是不是", "对吗", "咯", "吗", "呢",
    "关于", "相关", "内容", "这一类", "这类",
)

CONFIRMATION_BAD_CANDIDATES = {
    "这个", "那个", "这些", "那些",
    "东西", "方面", "类别", "分类", "情况",
    "比较", "就是", "还是", "应该", "感觉",
}

CANDIDATE_SUFFIXES = ("方面", "相关", "问题", "情况", "内容", "这块", "这一块")
BAD_CANDIDATE_FRAGMENTS = {"关系", "问题", "情况", "内容", "方面", "相关", "最近"}

SYSTEM_CAPABILITY_KEYWORDS = (
    "能干啥", "你能做什么", "能做什么", "可以做什么", "你可以做什么",
    "你的功能", "有什么功能", "有啥功能", "怎么用",

    "你能做啥", "能做啥", "做啥", "干啥",
)
JUDGMENT_LIKE_KEYWORDS = (
    "胡闹吗", "合理吗", "过分吗", "离谱吗",
    "谁的问题", "谁有问题", "算不算",
    "是不是", "对吗", "你怎么看", "你研究一下", "你判断一下",
)

EMOJI_RESPONSES = {"😁", "😄", "😆", "😂", "🤣", "😊", "😅", "🙂", "😉"}


def clean_text(text: str | None) -> str:
    return (text or "").strip().lower()


def contains_any(text: str | None, keywords: Iterable[str]) -> bool:
    normalized = clean_text(text)
    return any(keyword in normalized for keyword in keywords)


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def split_cn_text(text: str) -> list[str]:
    normalized = text
    for old in ("，", "。", "？"):
        normalized = normalized.replace(old, " ")
    return [part.strip() for part in normalized.split() if part.strip()]


def normalize_tag(tag: str) -> str:
    normalized = clean_text(tag)
    if not normalized or normalized in BAD_TAGS or len(normalized) < 2:
        return ""
    return normalized


def merge_similar_tags(tag_counter: Counter) -> Counter:
    merged = Counter()

    for tag, count in tag_counter.most_common():
        matched_key = None

        for existing in list(merged.keys()):
            if tag == existing or tag in existing or existing in tag:
                matched_key = existing
                break

        if matched_key is None:
            merged[tag] += count
            continue

        better = matched_key if len(matched_key) >= len(tag) else tag
        worse = tag if better == matched_key else matched_key
        merged[better] += merged[worse] + count
        if worse in merged:
            del merged[worse]

    return merged


def iter_record_tags(record: dict) -> set[str]:
    shadow_tags = record.get("shadow_tags", "") or ""
    normalized_tags = {
        normalize_tag(tag)
        for tag in shadow_tags.split()
        if normalize_tag(tag)
    }
    normalized_tags.discard("")
    return normalized_tags


def extract_fine_topics(repo_state):
    from collections import defaultdict

    tag_to_paths = defaultdict(list)

    for record in getattr(repo_state, "doc_records", []):
        path = record.get("path", "")
        tags = (record.get("shadow_tags") or "").split()

        for tag in tags:
            tag = tag.strip()
            if not tag:
                continue
            if len(tag) < 2:
                continue
            if len(tag) > 20:
                continue

            tag_to_paths[tag].append(path)

    results = []

    for tag, paths in tag_to_paths.items():
        results.append({
            "tag": tag,
            "count": len(paths),
            "paths": list(dict.fromkeys(paths)),
        })

    results.sort(key=lambda x: x["count"], reverse=True)

    return results


def validate_summary_output(result: str) -> list[str]:
    text = (result or "").strip()
    if not text:
        raise RuntimeError("本地模型返回为空")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_lines = [line for line in lines if line.startswith("- ")]
    if len(bullet_lines) < 2:
        raise RuntimeError(f"本地模型输出不符合预期: {text}")

    return bullet_lines[:6]


def summarize_topics_coarsely_with_local_llm(
    fine_topics,
    topic_summarizer: Callable[[str], str] | None,
) -> str:
    if not topic_summarizer:
        raise RuntimeError("未提供 topic_summarizer，本地模型概括不可用")

    top_topics = [item["tag"] for item in fine_topics[:12]]
    if not top_topics:
        raise RuntimeError("没有可用于概括的细标签")

    prompt = (
        "下面是个人知识库中出现频率较高的一批细标签。\n"
        "请把它们压缩概括成 4 到 6 个更大的内容方面。\n"
        "必须遵守：\n"
        "1. 只输出列表\n"
        "2. 每行必须以“- ”开头\n"
        "3. 不要写前言，不要解释，不要总结\n"
        "4. 不要逐条复述原始细标签\n"
        "5. 要把相近内容合并成更大的方面\n"
        "6. 用自然中文，不要太学术\n"
        "7. 表达像人在整理自己的文档，而不是写报告\n\n"
        "细标签：\n"
        + "\n".join(f"- {topic}" for topic in top_topics)
    )

    bullet_lines = validate_summary_output(topic_summarizer(prompt))
    return "按更大的方面看，当前知识库主要集中在这些板块：\n" + "\n".join(bullet_lines)

def format_bytes(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes}B"

    units = ["KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        value /= 1024.0
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f}{unit}"

    return f"{num_bytes}B"
