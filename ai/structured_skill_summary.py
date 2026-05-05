from __future__ import annotations

import re


def looks_like_structured_skill_summary_request(question: str) -> bool:
    text = str(question or "").strip()
    if not text:
        return False
    normalized = re.sub(r"[，。！？,.!?\s]+", "", text.lower())
    has_structured_intent = any(marker in text for marker in ("归纳", "整理", "总结", "分类", "分成", "做个表", "表格"))
    has_analysis_target = any(marker in text for marker in ("要求", "能力", "经验", "技能", "技术", "工程"))
    has_followup_scope = any(marker in normalized for marker in ("这些材料", "这些文件", "目前这些", "反复出现", "通常怎么写"))
    return has_structured_intent and has_analysis_target and has_followup_scope


def build_structured_skill_summary_materials(repo_state) -> str | None:
    records = _collect_records(repo_state)
    if not records:
        return None

    valid_records: list[tuple[str, str, list[str]]] = []
    empty_paths: list[str] = []

    for path, doc in records:
        cleaned = _clean_doc(doc)
        if not cleaned:
            empty_paths.append(path)
            continue
        snippets = _extract_evidence_snippets(cleaned)
        if not snippets:
            empty_paths.append(path)
            continue
        valid_records.append((path, cleaned, snippets))

    if not valid_records and not empty_paths:
        return None

    lines: list[str] = []
    lines.append("## 结果集材料")
    lines.append(f"- 文件总数: {len(records)}")
    lines.append(f"- 有效文本文件数: {len(valid_records)}")
    lines.append(f"- 未提取到有效文本文件数: {len(empty_paths)}")

    if empty_paths:
        lines.append("\n## 未提取到有效文本")
        lines.extend(f"- {path}" for path in empty_paths[:20])

    lines.append("\n## 文件证据摘录")
    for idx, (path, cleaned, snippets) in enumerate(valid_records[:14], start=1):
        lines.append(f"\n### 文件{idx}: {path}")
        lines.append(f"- 文本预览: {cleaned[:180]}")
        for snippet_idx, snippet in enumerate(snippets[:6], start=1):
            lines.append(f"- 摘录{snippet_idx}: {snippet}")

    return "\n".join(lines)


def summarize_structured_skill_summary_with_remote(
    *,
    materials_markdown: str,
    question: str,
    client,
    model_id: str,
    logger,
) -> str:
    if not materials_markdown.strip() or client is None:
        return materials_markdown

    prompt = (
        "下面是当前结果集文件的原始证据整理，请基于这些材料自由归纳并回答用户问题。\n"
        "你可以自由组织表达、自由命名板块、自由决定是否用表格、列表或自然段。\n"
        "如果你觉得合适，可以按高频、中频、加分项、低频/不确定来整理；如果你觉得别的组织方式更自然，也可以自行处理。\n"
        "只需要注意：如果某个文件没有有效文本，就写“未提取到有效文本”，不要写“未提供具体内容”。\n\n"
        f"用户问题：{question}\n\n"
        "当前结果集证据：\n"
        f"{materials_markdown}"
    )
    try:
        response = client.models.generate_content(model=model_id, contents=prompt)
    except Exception as exc:
        logger.warning(f"⚠️ [结构化能力归纳] 远程归纳失败，回退本地证据包 ({exc})")
        return materials_markdown

    text = str(getattr(response, "text", "") or "").strip()
    if not text:
        return materials_markdown
    return text


def _collect_records(repo_state) -> list[tuple[str, str]]:
    path_to_doc: dict[str, str] = {}
    repo_paths = list(getattr(repo_state, "paths", []) or [])
    repo_docs = list(getattr(repo_state, "docs", []) or [])
    for idx, raw_path in enumerate(repo_paths):
        if idx >= len(repo_docs):
            break
        path = str(raw_path or "").strip()
        if path:
            path_to_doc[path] = str(repo_docs[idx] or "")

    records: list[tuple[str, str]] = []
    for record in list(getattr(repo_state, "doc_records", []) or []):
        path = str(record.get("path", "") or "").strip()
        if not path:
            continue
        doc = str(record.get("doc", "") or path_to_doc.get(path, "") or "")
        records.append((path, doc))
    return records


def _clean_doc(doc: str) -> str:
    text = re.sub(r"\s+", " ", str(doc or "")).strip()
    if len(text) < 12:
        return ""
    return text[:4000]


def _extract_evidence_snippets(doc: str) -> list[str]:
    parts = re.split(r"[\n\r]+|[。；;]", doc)
    snippets: list[str] = []
    for part in parts:
        snippet = re.sub(r"\s+", " ", part).strip(" -\t")
        if len(snippet) < 10:
            continue
        if any(marker in snippet for marker in ("职责", "要求", "熟悉", "掌握", "精通", "了解", "具备", "经验", "优先", "项目", "落地")):
            snippets.append(snippet[:220])
            continue
        ascii_tokens = re.findall(r"[A-Za-z][A-Za-z0-9+#./_-]{1,30}", snippet)
        if len(ascii_tokens) >= 2:
            snippets.append(snippet[:220])

    deduped: list[str] = []
    seen: set[str] = set()
    for snippet in snippets:
        key = snippet.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(snippet)
    return deduped[:12]
