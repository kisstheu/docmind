from __future__ import annotations

from pathlib import Path
import re
from typing import Callable, Sequence

from ai.capability_common import extract_fine_topics, summarize_topics_coarsely_with_local_llm
from ai.repo_meta.category.overview import _build_scene_category_summary, _extract_scene_topics


def answer_repo_content_category_question(repo_state) -> str:
    fine_topics = extract_fine_topics(repo_state)
    lines = [f"- {item['tag']}：约 {item['count']} 个文件" for item in fine_topics[:12]]
    return "按内容标签粗略来看，当前知识库主要集中在这些方面：\n" + "\n".join(lines) + "\n\n这是基于影子标签自动归纳出来的。"


def answer_repo_content_type_theme_summary_question(repo_state, topic_summarizer) -> str | None:
    if not topic_summarizer:
        return None

    body_excerpts = _collect_body_excerpts(repo_state, limit=8, excerpt_limit=220)
    if len(body_excerpts) < 2:
        return None

    prompt = (
        "下面是一组待归纳材料的内容片段。\n"
        "请根据这些内容，归纳它们整体的“材料类型或用途 + 共同主题”。\n"
        "必须遵守：\n"
        "1. 先识别这些材料在现实中的类型、用途或场景，再概括反复出现的主题\n"
        "2. 只能依据所给内容，不得依据文件名、索引标签或预设领域猜测\n"
        "3. 只输出一到两句自然中文，不要列表、标题、解释或逐项复述\n"
        "4. 若内容不足以判断具体类型，应保守描述，不得编造\n\n"
        "内容样本：\n"
        + "\n".join(
            f"[样本 {idx}] {excerpt}"
            for idx, excerpt in enumerate(body_excerpts, start=1)
        )
    )

    try:
        raw = topic_summarizer(prompt)
    except Exception:
        return None
    return _normalize_type_theme_summary(raw)


def _collect_body_excerpts(repo_state, *, limit: int, excerpt_limit: int) -> list[str]:
    repo_paths = [str(path or "").strip() for path in list(getattr(repo_state, "paths", []) or [])]
    repo_docs = list(getattr(repo_state, "docs", []) or [])
    body_by_path = {
        path: str(repo_docs[idx] or "")
        for idx, path in enumerate(repo_paths)
        if path and idx < len(repo_docs)
    }
    for record in list(getattr(repo_state, "doc_records", []) or []):
        path = str(record.get("path", "") or "").strip()
        body = str(record.get("doc", "") or "").strip()
        if path and body:
            body_by_path[path] = body

    bodies = [body_by_path[path] for path in repo_paths if body_by_path.get(path)]
    if not bodies:
        bodies = [
            str(record.get("doc", "") or "").strip()
            for record in list(getattr(repo_state, "doc_records", []) or [])
            if str(record.get("doc", "") or "").strip()
        ]
    if not bodies:
        return []

    if len(bodies) <= limit:
        selected = bodies
    elif limit <= 1:
        selected = [bodies[0]]
    else:
        last_index = len(bodies) - 1
        selected_indices = sorted({round(i * last_index / float(limit - 1)) for i in range(limit)})
        selected = [bodies[idx] for idx in selected_indices]

    excerpts: list[str] = []
    for body in selected:
        excerpt = re.sub(r"\s+", " ", str(body or "")).strip(" ，,。；;")
        if len(excerpt) < 12:
            continue
        excerpts.append(excerpt[:excerpt_limit].strip(" ，,。；;"))
    return excerpts


def _normalize_type_theme_summary(raw_text: str) -> str | None:
    lines: list[str] = []
    for line in str(raw_text or "").splitlines():
        text = line.strip()
        if not text or text.startswith("```"):
            continue
        text = re.sub(r"^(?:[-*•]|\d+[.、])\s*", "", text).strip()
        text = re.sub(r"^(?:文档类型|材料类型|共同主题|总体|整体)\s*[:：]\s*", "", text).strip()
        if text:
            lines.append(text)
        if len(lines) >= 2:
            break
    if not lines:
        return None

    summary = "，".join(lines)
    summary = re.sub(r"\s+", " ", summary).strip(" ，,。；;")
    if not summary:
        return None
    return summary if summary.endswith(("。", "！", "？")) else f"{summary}。"


def answer_repo_content_category_summary_question(repo_state, topic_summarizer) -> str:
    tag_guided_summary = _build_tag_guided_category_summary(
        repo_state,
        topic_summarizer=topic_summarizer,
    )
    if tag_guided_summary:
        return tag_guided_summary

    excerpt_summary = _build_excerpt_based_category_summary(
        repo_state,
        topic_summarizer=topic_summarizer,
    )
    if excerpt_summary:
        return excerpt_summary

    scene_summary = _build_scene_category_summary(repo_state)
    if scene_summary:
        return scene_summary

    scene_topics = _extract_scene_topics(repo_state)
    if scene_topics:
        try:
            return summarize_topics_coarsely_with_local_llm(
                fine_topics=scene_topics,
                topic_summarizer=topic_summarizer,
                topic_source="场景标签",
                prefer_scene=True,
            )
        except Exception:
            pass

    fine_topics = extract_fine_topics(repo_state)
    if not fine_topics:
        return "当前标签信息不足，暂时还无法按更大的方面概括。"

    return summarize_topics_coarsely_with_local_llm(
        fine_topics=fine_topics,
        topic_summarizer=topic_summarizer,
        topic_source="细标签",
    )


def _build_tag_guided_category_summary(
    repo_state,
    topic_summarizer: Callable[[str], str] | None,
) -> str | None:
    tag_topics = _extract_tag_guided_topics(repo_state)
    if not tag_topics:
        return None

    # 标签已经是建库阶段的压缩信号，优先让本地模型沿这些主线做更高层归并。
    if topic_summarizer:
        try:
            return summarize_topics_coarsely_with_local_llm(
                fine_topics=tag_topics,
                topic_summarizer=topic_summarizer,
                topic_source="建库标签",
                prefer_scene=True,
            )
        except Exception:
            pass

    selected = [str(item.get("tag", "") or "").strip() for item in tag_topics[:5] if str(item.get("tag", "") or "").strip()]
    if len(selected) < 2:
        return None
    return "按更大的方面看，当前知识库主要集中在这些板块：\n" + "\n".join(f"- {tag}" for tag in selected)


def _build_excerpt_based_category_summary(
    repo_state,
    topic_summarizer: Callable[[str], str] | None,
) -> str | None:
    if not topic_summarizer:
        return None

    records = list(getattr(repo_state, "doc_records", []) or [])
    if not records or len(records) > 24:
        return None

    path_to_doc: dict[str, str] = {}
    repo_paths = list(getattr(repo_state, "paths", []) or [])
    repo_docs = list(getattr(repo_state, "docs", []) or [])
    for idx, path in enumerate(repo_paths):
        if idx >= len(repo_docs):
            break
        clean_path = str(path or "").strip()
        if clean_path:
            path_to_doc[clean_path] = str(repo_docs[idx] or "")

    excerpt_lines: list[str] = []
    for record in records[:12]:
        path = str(record.get("path", "") or "").strip()
        doc = str(record.get("doc", "") or path_to_doc.get(path, "") or "").strip()
        if not path or not doc:
            continue

        excerpt = _extract_thematic_excerpt(doc)
        if not excerpt:
            continue
        excerpt_lines.append(f"- {Path(path).name}: {excerpt}")

    if len(excerpt_lines) < 3:
        return None

    prompt = (
        "下面是一组知识库文件的摘录。\n"
        "请先判断它们整体更像什么资料或问题场景，再给出 3 到 5 个较大的内容板块。\n"
        "必须遵守：\n"
        "1. 只输出列表\n"
        "2. 第一行必须以“- 整体：”开头，用一句自然中文概括这批资料的主线\n"
        "3. 后续每行必须以“- ”开头，输出 3 到 5 个板块\n"
        "4. 优先概括资料类型、用途场景或重复出现的任务主题，不要只罗列零散技术词\n"
        "5. 只能依据摘录里反复出现的信息，不要编造未出现的结论\n\n"
        "文件摘录：\n"
        + "\n".join(excerpt_lines)
    )

    try:
        raw = topic_summarizer(prompt)
    except Exception:
        raw = ""

    summary = _format_excerpt_summary_from_raw(raw)
    if summary:
        return summary

    return _build_keyword_based_excerpt_summary(excerpt_lines)


def _normalize_summary_bullet_line(line: str) -> str:
    text = str(line or "").strip()
    if not text.startswith("- "):
        return text
    label = text[2:].strip()
    label = re.sub(r"^[-*•]+\s*", "", label)
    return f"- {label}" if label else text


def _extract_thematic_excerpt(doc: str) -> str:
    text = str(doc or "").strip()
    if not text:
        return ""

    excerpt = re.sub(r"\s+", " ", text.replace("\r\n", "\n").replace("\r", "\n"))
    excerpt = excerpt[:220].strip(" ,，。；;")
    return excerpt


def _extract_tag_guided_topics(repo_state) -> list[dict]:
    path_topics: dict[str, set[str]] = {}
    for record in list(getattr(repo_state, "doc_records", []) or []):
        path = str(record.get("path", "") or "").strip()
        if not path:
            continue
        for token in _iter_tag_summary_tokens(record):
            path_topics.setdefault(token, set()).add(path)

    results = [
        {"tag": tag, "count": len(paths), "paths": sorted(paths)}
        for tag, paths in path_topics.items()
    ]
    results.sort(key=lambda item: (-int(item.get("count", 0) or 0), -len(str(item.get("tag", "") or "")), str(item.get("tag", "") or "")))
    return results


def _iter_tag_summary_tokens(record: dict) -> list[str]:
    scene_tokens = _split_summary_tag_tokens(record.get("scene_tags", ""))
    shadow_tokens = _split_summary_tag_tokens(record.get("shadow_tags", ""))

    weighted_tokens = shadow_tokens + shadow_tokens + scene_tokens
    deduped: list[str] = []
    seen: set[str] = set()
    for token in weighted_tokens:
        if token in seen:
            continue
        seen.add(token)
        deduped.append(token)
    return deduped


def _split_summary_tag_tokens(raw_text: str) -> list[str]:
    tokens: list[str] = []
    for token in re.split(r"[\s,，、;；|/]+", str(raw_text or "").strip()):
        cleaned = str(token or "").strip()
        if not cleaned:
            continue
        tokens.append(cleaned)
    return tokens


def _format_excerpt_summary_from_raw(raw: str) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None

    normalized_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^[-*•]\s*", stripped):
            stripped = re.sub(r"^[-*•]\s*", "- ", stripped)
        normalized_lines.append(stripped)

    if not normalized_lines:
        return None

    overall_line = ""
    category_lines: list[str] = []
    for line in normalized_lines:
        compact = re.sub(r"^-\s*", "", line).strip()
        if not compact:
            continue
        if re.match(r"^(整体|总体|主线|主题)[:：]", compact):
            overall_line = "- 整体：" + re.split(r"[:：]", compact, maxsplit=1)[1].strip()
            continue
        if line.startswith("- "):
            category_lines.append(_normalize_summary_bullet_line(line))

    if not overall_line and normalized_lines:
        first_line = normalized_lines[0]
        compact = re.sub(r"^-\s*", "", first_line).strip()
        looks_like_sentence = (
            6 <= len(compact) <= 40
            and not compact.endswith(("：", ":"))
            and any(marker in compact for marker in ("这批", "这些", "资料", "材料", "文件", "文档", "主要", "围绕"))
        )
        if looks_like_sentence:
            overall_line = f"- 整体：{compact}"
            if first_line.startswith("- "):
                category_lines = category_lines[1:]

    deduped_categories: list[str] = []
    seen: set[str] = set()
    for line in category_lines:
        if line in seen:
            continue
        seen.add(line)
        deduped_categories.append(line)

    if not overall_line or len(deduped_categories) < 2:
        return None

    return overall_line[2:].strip() + "\n" + "\n".join(deduped_categories[:5])


def _build_keyword_based_excerpt_summary(excerpt_lines: Sequence[str]) -> str | None:
    if len(excerpt_lines) < 3:
        return None

    token_to_files: dict[str, set[int]] = {}
    for idx, line in enumerate(excerpt_lines):
        body = line.split(":", 1)[-1]
        tokens = set(_iter_excerpt_keywords(body))
        for token in tokens:
            token_to_files.setdefault(token, set()).add(idx)

    min_count = 2 if len(excerpt_lines) <= 6 else 3
    ranked = [
        (token, len(file_ids), _score_excerpt_token(token, file_ids))
        for token, file_ids in token_to_files.items()
        if len(file_ids) >= min_count
    ]
    ranked.sort(key=lambda item: (-item[2], -item[1], -len(item[0]), item[0]))
    if len(ranked) < 3:
        return None

    top_terms = _prune_redundant_excerpt_terms(ranked)[:5]
    if len(top_terms) < 3:
        return None

    overall_terms = "、".join(top_terms[:3])
    lines = [f"- {token}" for token in top_terms]
    return f"整体：这批资料主要围绕 {overall_terms} 等重复主题\n" + "\n".join(lines)


def _iter_excerpt_keywords(text: str) -> list[str]:
    patterns = re.findall(r"[A-Za-z][A-Za-z0-9_+.#/-]{1,31}|[\u4e00-\u9fff]{2,8}", text or "")
    keywords: list[str] = []
    for token in patterns:
        normalized = token.strip()
        if not normalized:
            continue
        candidate = _normalize_excerpt_candidate(normalized)
        if not _is_meaningful_excerpt_candidate(candidate):
            continue
        keywords.append(candidate)
    return keywords


def _normalize_excerpt_candidate(token: str) -> str:
    normalized = str(token or "").strip()
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+.#/-]{1,31}", normalized):
        return normalized
    return re.sub(r"\s+", "", normalized)


def _is_meaningful_excerpt_candidate(token: str) -> bool:
    if not token:
        return False
    if token.isdigit():
        return False
    if len(set(token)) == 1:
        return False
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+.#/-]{1,31}", token):
        return any(ch.isalpha() for ch in token)
    if not re.fullmatch(r"[\u4e00-\u9fff]{2,8}", token):
        return False
    if len(token) <= 2:
        return False

    generic_tokens = {
        "这个", "那个", "这些", "那些", "一种", "一些", "以及", "因为", "所以", "如果",
        "可以", "需要", "进行", "包括", "相关", "其中", "以及", "我们", "你们", "他们",
        "当前", "目前", "这里", "那里", "内容", "资料", "文件", "文档", "整体", "主要",
        "主题", "问题", "情况", "方面", "东西", "部分", "能力", "经验", "要求",
    }
    if token in generic_tokens:
        return False

    function_chars = set("的了和及与在对将把等并或而且中上下内外其所")
    non_function_chars = [ch for ch in token if ch not in function_chars]
    if len(non_function_chars) < 2:
        return False
    if token[0] in function_chars and token[-1] in function_chars:
        return False
    return True


def _score_excerpt_token(token: str, file_ids: set[int]) -> float:
    coverage = float(len(file_ids))
    diversity = float(len(set(token)))
    english_bonus = 0.5 if re.search(r"[A-Za-z]", token) else 0.0
    return coverage * 10.0 + min(diversity, 6.0) + english_bonus + (len(token) * 0.1)


def _prune_redundant_excerpt_terms(
    ranked: Sequence[tuple[str, int, float]],
) -> list[str]:
    selected: list[str] = []
    for token, _count, _score in ranked:
        if any(token in existing or existing in token for existing in selected):
            continue
        selected.append(token)
    return selected
