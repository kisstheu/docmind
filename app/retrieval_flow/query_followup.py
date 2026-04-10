from __future__ import annotations

import re

from app.chat_text_utils import (
    extract_strong_terms_from_question,
    is_result_expansion_followup,
    normalize_question_for_retrieval,
)

_FORCE_WIDER_SCOPE_MARKERS = (
    "扩大范围",
    "范围大点",
    "范围放宽",
    "放宽范围",
    "放宽一点",
    "扩大检索",
    "全库",
    "全局",
    "别只看",
)

GENERIC_FOLLOWUP_MARKERS = (
    "更多",
    "继续",
    "还有",
    "再来",
    "补充",
    "接着",
    "然后",
    "后续",
)

_ANALYTIC_FOLLOWUP_MARKERS = (
    "这些",
    "那些",
    "其中",
    "哪一类",
    "哪类",
    "需求最多",
    "数量",
    "占比",
    "简单说明",
    "按数量",
    "按占比",
)


def _wants_force_wider_scope(question: str) -> bool:
    q = normalize_question_for_retrieval(question)
    if not q:
        return False
    return any(marker in q for marker in _FORCE_WIDER_SCOPE_MARKERS)


def should_reuse_previous_results(question: str, event, last_relevant_indices) -> bool:
    if _wants_force_wider_scope(question):
        return False
    return (
        getattr(event, "name", "") == "content_followup"
        and bool(last_relevant_indices)
        and is_result_expansion_followup(question)
    )


def filter_reused_indices_for_question(question: str, candidate_indices, repo_state, logger):
    q = normalize_question_for_retrieval(question)
    if not q:
        return list(candidate_indices)

    focus_terms = extract_strong_terms_from_question(q)
    if not focus_terms:
        return list(candidate_indices)

    scored = []
    for idx in candidate_indices:
        path = repo_state.chunk_paths[idx]
        text = repo_state.chunk_texts[idx] or ""
        score = 0
        for term in focus_terms:
            if term and term in text:
                score += 2
            elif term and term in path:
                score += 1

        if "非法" in path and "非法" not in text:
            score -= 2
        if "违法" in path and "违法" not in text:
            score -= 2
        scored.append((score, idx))

    scored.sort(reverse=True, key=lambda x: x[0])
    filtered = [idx for score, idx in scored if score > 0]
    if filtered:
        logger.info(f"🔍 [追问结果细筛] focus_terms={focus_terms}")
        logger.info(f"🔍 [追问结果细筛] 保留 {len(filtered)} / {len(candidate_indices)} 个片段")
        return filtered[:12]
    return list(candidate_indices)


def _looks_like_contextual_analytic_followup(question: str) -> bool:
    q = normalize_question_for_retrieval(question)
    if not q:
        return False
    if any(marker in q for marker in _ANALYTIC_FOLLOWUP_MARKERS):
        return True
    if re.search(r"这些.*方向", q):
        return True
    if re.search(r"哪.*类.*最多", q):
        return True
    return False


def should_keep_followup_anchor(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    if _looks_like_contextual_analytic_followup(q):
        return True

    weak_followups = {
        "我能赢吗",
        "能赢吗",
        "能不能赢",
        "有胜算吗",
        "胜算大吗",
        "赢面大吗",
        "把握大吗",
        "合理吗",
        "合法吗",
        "过分吗",
        "怎么办",
        "会怎样",
        "结果吗",
        "然后呢",
        "后来吗",
        "为什么",
        "依据是什么",
        "有哪些证据",
    }
    if q in weak_followups:
        return True
    return len(q) <= 12


def _is_generic_followup_question(question: str) -> bool:
    q = normalize_question_for_retrieval(question)
    if not q:
        return False
    if q in {"更多", "继续", "还有", "再来", "补充"}:
        return True
    if len(q) <= 6 and any(marker in q for marker in GENERIC_FOLLOWUP_MARKERS):
        return True
    return False


def _collapse_duplicate_tail_tokens(merged_query: str, question: str) -> str:
    merged = (merged_query or "").strip()
    q = (question or "").strip()
    if not merged or not q:
        return merged

    merged_tokens = [x for x in merged.split() if x.strip()]
    if len(merged_tokens) < 2:
        return merged

    q_norm = normalize_question_for_retrieval(q)
    if not q_norm:
        return merged

    if (
        normalize_question_for_retrieval(merged_tokens[-1]) == q_norm
        and normalize_question_for_retrieval(merged_tokens[-2]) == q_norm
    ):
        return " ".join(merged_tokens[:-1]).strip()
    return merged


def _stabilize_followup_merged_query(
    *,
    merged_query: str,
    question: str,
    last_effective_search_query: str | None,
    logger=None,
) -> str:
    merged = (merged_query or "").strip()
    q = (question or "").strip()
    if not merged:
        return merged

    merged = _collapse_duplicate_tail_tokens(merged, q)
    if not _is_generic_followup_question(q):
        return merged

    last_query = normalize_question_for_retrieval(last_effective_search_query or "")
    if not last_query:
        return merged

    q_norm = normalize_question_for_retrieval(q)
    if q_norm and last_query.endswith(q_norm):
        stabilized = last_query
    else:
        stabilized = f"{last_query} {q_norm}".strip() if q_norm else last_query

    if logger and stabilized != merged:
        logger.info(f"🔧 [追问继承修正] {merged} -> {stabilized}")
    return stabilized


def _force_company_name_anchor_for_followup(
    *,
    search_query: str,
    base_query: str,
    question: str,
    last_answer_type: str | None,
    logger=None,
) -> str:
    q = (search_query or "").strip() or (base_query or "").strip()
    if last_answer_type != "enumeration_company":
        return q
    if not _is_generic_followup_question(question):
        return q

    combined = f"{q} {base_query or ''}".replace(" ", "")
    has_company_signal = any(token in combined for token in ("公司", "企业", "单位", "组织"))
    if not has_company_signal:
        return q

    terms = [x for x in q.split() if x.strip()]
    changed = False
    for anchor in ("公司", "名称"):
        if anchor not in terms:
            terms.append(anchor)
            changed = True

    result = " ".join(terms).strip()
    if changed and logger:
        logger.info(f"🔧 [公司名追问保锚] {q} -> {result}")
    return result


def _is_global_timeline_request(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    has_timeline_intent = any(x in q for x in ["时间线", "按时间", "按日期", "按时间顺序"])
    has_global_scope = any(x in q for x in ["所有", "全部", "最近所有", "整体", "全局", "总体"])
    return has_timeline_intent and has_global_scope


def _build_global_timeline_query() -> str:
    terms = [
        "最近",
        "时间线",
        "记录",
        "会议纪要",
        "学习笔记",
        "周报",
        "复盘",
        "项目计划",
        "任务看板",
        "访谈",
        "生活",
    ]
    return " ".join(terms)
