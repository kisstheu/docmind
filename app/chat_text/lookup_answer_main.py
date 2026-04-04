from __future__ import annotations

from app.chat_text.lookup_common import (
    DIRECT_LOOKUP_GENERIC_ANCHOR_TERMS,
    _extract_selector_signatures,
    _normalize_lookup_token,
)
from app.chat_text.lookup_predicates import (
    _expand_role_terms,
    _is_role_like_term,
    _looks_like_company_hr_mapping_query,
    _looks_like_mapping_followup_query,
    _looks_like_role_name_query,
)
from app.chat_text.lookup_extract_company import _extract_company_hr_mapping_items
from app.chat_text.lookup_extract_role import _extract_role_name_items
from app.chat_text.lookup_answer_helpers import (
    _build_direct_lookup_evidence_items,
    _extract_direct_lookup_focus_terms,
    _extract_direct_lookup_terms,
    _looks_like_direct_lookup_followup_question,
    _looks_like_direct_lookup_question,
)

def maybe_build_direct_lookup_answer(
    *,
    question: str,
    search_query: str,
    relevant_indices,
    repo_state,
    max_items: int = 6,
    logger=None,
    allow_followup_inference: bool = False,
    force_local_evidence: bool = False,
) -> str | None:
    focus_terms = _extract_direct_lookup_focus_terms(question)
    is_role_name_query, _ = _looks_like_role_name_query(question, focus_terms)
    if (not force_local_evidence) and (not _looks_like_direct_lookup_question(question)):
        if not is_role_name_query:
            if not allow_followup_inference:
                return None
            if not _looks_like_direct_lookup_followup_question(question):
                return None

    terms = _extract_direct_lookup_terms(question, search_query)
    if not terms:
        return None
    anchor_terms: list[str] = []
    for token in terms:
        norm = _normalize_lookup_token(token)
        if not norm:
            continue
        if _is_role_like_term(norm):
            continue
        if norm in {"公司", "企业", "单位"}:
            continue
        if norm in DIRECT_LOOKUP_GENERIC_ANCHOR_TERMS:
            continue
        if token not in anchor_terms:
            anchor_terms.append(token)

    is_company_hr_mapping_query, role_terms_for_mapping = _looks_like_company_hr_mapping_query(question, focus_terms)
    if is_company_hr_mapping_query:
        role_terms_for_mapping = _expand_role_terms(role_terms_for_mapping)
        required_selector_signatures = _extract_selector_signatures(question)
        if not required_selector_signatures:
            required_selector_signatures = _extract_selector_signatures(search_query)
        mapping_items = _extract_company_hr_mapping_items(
            role_terms=role_terms_for_mapping,
            anchor_terms=anchor_terms,
            required_selector_signatures=required_selector_signatures,
            relevant_indices=relevant_indices,
            repo_state=repo_state,
            max_items=max_items,
        )
        if mapping_items:
            lines = ["根据当前检索片段，匹配到以下“公司-HR”对应关系："]
            for i, item in enumerate(mapping_items, start=1):
                lines.append(f"{i}. {item['company']}（HR：{item['name']}）")
                if item["role_line"]:
                    lines.append(f"   关联线索：{item['role_line']}")
                lines.append(f"   来源：{item['path']}")
            if logger:
                logger.info(f"🧪 [直接检索稳态回答] 公司-HR对应命中 {len(mapping_items)} 条")
            return "\n".join(lines)

    if _looks_like_mapping_followup_query(
        question,
        focus_terms,
        allow_followup_inference=allow_followup_inference,
    ):
        inferred_role_terms = _expand_role_terms(
            [token for token in focus_terms if _is_role_like_term(_normalize_lookup_token(token))]
        )
        required_selector_signatures = _extract_selector_signatures(question)
        if not required_selector_signatures:
            required_selector_signatures = _extract_selector_signatures(search_query)
        mapping_items = _extract_company_hr_mapping_items(
            role_terms=inferred_role_terms,
            anchor_terms=anchor_terms,
            required_selector_signatures=required_selector_signatures,
            relevant_indices=relevant_indices,
            repo_state=repo_state,
            max_items=max_items,
        )
        if mapping_items:
            lines = ["根据当前检索片段，按上一轮口径匹配到以下“公司-HR”对应关系："]
            for i, item in enumerate(mapping_items, start=1):
                lines.append(f"{i}. {item['company']}（HR：{item['name']}）")
                if item["role_line"]:
                    lines.append(f"   关联线索：{item['role_line']}")
                lines.append(f"   来源：{item['path']}")
            if logger:
                logger.info(f"🧪 [直接检索稳态回答] 追问口径继承-公司HR命中 {len(mapping_items)} 条")
            return "\n".join(lines)

    is_role_name_query, role_terms = _looks_like_role_name_query(question, focus_terms)
    if is_role_name_query:
        role_terms = _expand_role_terms(role_terms)
        required_selector_signatures = _extract_selector_signatures(question)
        if not required_selector_signatures:
            required_selector_signatures = _extract_selector_signatures(search_query)
        q_norm = _normalize_lookup_token(question)
        role_max_items = max_items
        is_singular_owner_query = (
            ("的" in (question or ""))
            and not any(marker in (question or "") for marker in ("哪些", "哪几", "分别", "列表", "清单", "和", "以及", "及", "/", "、"))
        )
        if is_singular_owner_query:
            role_max_items = 1
        if allow_followup_inference and len(q_norm) <= 8:
            role_max_items = min(max_items, 4)
        role_name_items = _extract_role_name_items(
            role_terms=role_terms,
            anchor_terms=anchor_terms,
            required_selector_signatures=required_selector_signatures,
            relevant_indices=relevant_indices,
            repo_state=repo_state,
            max_items=role_max_items,
        )
        if not role_name_items and anchor_terms:
            role_name_items = _extract_role_name_items(
                role_terms=role_terms,
                anchor_terms=[],
                required_selector_signatures=required_selector_signatures,
                relevant_indices=relevant_indices,
                repo_state=repo_state,
                max_items=role_max_items,
            )
        if role_name_items:
            role_label = " / ".join(role_terms[:2])
            lines = [f"根据当前检索片段，匹配到 {role_label} 相关姓名："]
            for i, item in enumerate(role_name_items, start=1):
                if item.get("company"):
                    lines.append(f"{i}. {item['name']}（{item['company']}）")
                else:
                    lines.append(f"{i}. {item['name']}")
                lines.append(f"   来源：{item['path']}")
                lines.append(f"   证据：{item['evidence']}")
            if logger:
                logger.info(f"🧪 [直接检索稳态回答] 角色姓名命中 {len(role_name_items)} 条")
            return "\n".join(lines)

    items = _build_direct_lookup_evidence_items(
        terms=terms,
        focus_terms=focus_terms,
        relevant_indices=relevant_indices,
        repo_state=repo_state,
        max_items=max_items,
    )
    if not items:
        if force_local_evidence:
            return "根据当前检索片段，暂未提取到稳定的可核对条目。可继续说“看下1/看下2”查看来源文件。"
        if allow_followup_inference and focus_terms:
            focus_tip = "、".join(focus_terms[:3])
            return f"当前检索片段未直接命中“{focus_tip}”相关证据，先不给出推断；可补充更完整关键词后再查。"
        return None

    lines = ["根据当前检索片段，先给你可直接核对的证据："]
    for i, item in enumerate(items, start=1):
        lines.append(f"{i}. {item['line']}")
        lines.append(f"   来源：{item['path']}")

    if logger:
        logger.info(f"🧪 [直接检索稳态回答] 命中 {len(items)} 条证据")

    return "\n".join(lines)


FILE_LOOKUP_POLITE_PREFIXES = (
    "帮我查下",
    "帮我查找",
    "帮我查询",
    "帮我找下",
    "帮我看看",
    "帮我看下",
    "帮我定位下",
    "帮我定位一下",
    "帮我",
    "麻烦你",
    "麻烦",
    "请你",
    "请",
    "查下",
    "查找",
    "查询",
    "看看",
    "看下",
    "定位下",
    "定位一下",
)

FILE_LOOKUP_GENERIC_TERMS = {
    "文件",
    "文档",
    "记录",
    "哪个",
    "哪些",
    "哪份",
    "哪一份",
    "在哪",
    "在哪个",
    "是在",
    "是在哪",
    "文件名",
    "文档名",
    "记录名",
    "帮我",
    "查下",
    "查找",
    "查询",
    "看看",
    "看下",
}

FILE_LOOKUP_FOLLOWUP_PRONOUNS = {
    "这",
    "那",
    "这个",
    "那个",
    "它",
    "他",
    "她",
    "它们",
    "他们",
    "这条",
    "那条",
    "这个岗位",
    "那个岗位",
    "这个职位",
    "那个职位",
    "这个公司",
    "那个公司",
}
