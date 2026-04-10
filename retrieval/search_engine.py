from __future__ import annotations

import datetime
import os
import re
from pathlib import Path

import numpy as np

from retrieval.query_utils import (
    EXTENSION_TERMS,
    extract_query_terms,
)
from retrieval.search_context import build_context_text, build_inventory_candidates_text
from retrieval.search_intent import (
    determine_query_flags,
    is_company_name_lookup_context,
    is_compare_intent_query,
    is_entity_name_lookup_context,
    is_expansion_followup_question,
    is_file_location_lookup_query,
    is_relation_mismatch_query,
    is_related_record_listing_query,
    is_weak_query,
    rescue_entity_lookup_indices,
    should_enable_fallback,
)
from retrieval.search_term_scoring import (
    build_company_candidate_df,
    company_hint_bonus,
    is_result_set_boilerplate_term,
    is_sentence_like_term,
    should_score_body_term,
    should_score_filename_term,
)


def perform_retrieval(
    question: str,
    search_query: str,
    repo_state,
    model_emb,
    logger,
    current_focus_file: str | None,
    context_anchor: str = "",
    allowed_paths=None,
    scope_label: str | None = None,
):
    chunk_texts = list(getattr(repo_state, "chunk_texts", []) or [])
    chunk_paths = list(getattr(repo_state, "chunk_paths", []) or [])
    chunk_file_times = list(getattr(repo_state, "chunk_file_times", []) or [])
    chunk_embeddings = np.asarray(getattr(repo_state, "chunk_embeddings", np.empty((0,))))

    if not chunk_texts or chunk_embeddings.size == 0:
        logger.info("   [retrieval skipped] no indexed chunks available")
        return {"relevant_indices": [], "scores": np.empty((0,)), "current_focus_file": current_focus_file, }

    if chunk_embeddings.ndim == 1:
        chunk_embeddings = chunk_embeddings.reshape(1, -1)

    if chunk_embeddings.ndim != 2:
        logger.warning(f"   [retrieval skipped] invalid chunk embedding shape: {chunk_embeddings.shape}")
        return {"relevant_indices": [], "scores": np.empty((0,)), "current_focus_file": current_focus_file, }

    max_count = min(len(chunk_texts), len(chunk_paths), len(chunk_file_times), int(chunk_embeddings.shape[0]))
    if max_count <= 0:
        logger.info("   [retrieval skipped] inconsistent empty retrieval state")
        return {"relevant_indices": [], "scores": np.empty((0,)), "current_focus_file": current_focus_file, }

    if max_count < len(chunk_texts) or max_count < int(chunk_embeddings.shape[0]):
        logger.warning(
            f"   [retrieval alignment] chunk_texts={len(chunk_texts)}, "
            f"chunk_paths={len(chunk_paths)}, chunk_file_times={len(chunk_file_times)}, "
            f"chunk_embeddings_rows={int(chunk_embeddings.shape[0])}; using first {max_count}"
        )

    chunk_texts = chunk_texts[:max_count]
    chunk_paths = chunk_paths[:max_count]
    chunk_file_times = chunk_file_times[:max_count]
    chunk_embeddings = chunk_embeddings[:max_count]

    allowed_path_set = None
    candidate_indices = list(range(max_count))
    if allowed_paths is not None:
        allowed_path_set = {
            str(path or "").strip()
            for path in allowed_paths
            if str(path or "").strip()
        }
        candidate_indices = [idx for idx, path in enumerate(chunk_paths) if path in allowed_path_set]
        if current_focus_file and current_focus_file not in allowed_path_set:
            current_focus_file = None
        if not candidate_indices:
            scope_name = (scope_label or "指定范围").strip() or "指定范围"
            logger.info(f"   [范围约束未命中] {scope_name} 内暂无可检索片段")
            return {"relevant_indices": [], "scores": np.empty((0,)), "current_focus_file": current_focus_file, }
        scope_name = (scope_label or "指定范围").strip() or "指定范围"
        scoped_paths = {chunk_paths[idx] for idx in candidate_indices}
        logger.info(f"   [范围约束] {scope_name} -> {len(scoped_paths)} files / {len(candidate_indices)} chunks")
    candidate_index_set = set(candidate_indices)

    q_emb = model_emb.encode(["为这个句子生成表示以用于检索相关文章：" + search_query])[0]
    scores = np.dot(chunk_embeddings, q_emb)

    if context_anchor.strip():
        anchor_emb = model_emb.encode(["为这个句子生成表示以用于检索相关文章：" + context_anchor])[0]
        anchor_scores = np.dot(chunk_embeddings, anchor_emb)
        scores = scores + 0.12 * anchor_scores
        logger.info(f"   🪝 [锚点辅助检索] {context_anchor}")

    raw_search_terms = extract_query_terms(search_query, question)

    search_terms = []
    for term in raw_search_terms:
        t = (term or "").strip()
        if not t:
            continue

        # 先去掉明显整句型垃圾
        if is_sentence_like_term(t):
            logger.debug(f"   🚫 [过滤整句词] '{t}'")
            continue

        if is_result_set_boilerplate_term(t):
            logger.debug(f"   🚫 [过滤模板词] '{t}'")
            continue

        # 去重保序
        if t not in search_terms:
            search_terms.append(t)

    # 关系/不一致类问题：强制补充关系词，避免只剩“内容 文件名”
    if is_relation_mismatch_query(question):
        relation_terms = ["不一致", "不匹配", "不符", "对不上", "冲突", "矛盾"]
        anchor_terms = ["文件名", "标题", "正文", "内容", "名称"]
        for term in relation_terms + anchor_terms:
            if term not in search_terms:
                search_terms.append(term)
        logger.debug(f"   🧭 [关系型问题补词] {search_terms}")

    logger.debug(f"提取到的核心搜索词: {search_terms}")

    is_entity_lookup = is_entity_name_lookup_context(question, search_terms)
    is_company_lookup = is_company_name_lookup_context(question, search_terms)
    is_expansion_followup = is_expansion_followup_question(question)
    company_candidate_df: dict[str, int] | None = None
    company_docs_total = 1
    if is_company_lookup:
        company_candidate_df, company_docs_total = build_company_candidate_df(
            list(getattr(repo_state, "docs", []) or [])
        )

    try:
        max_match_log_lines = int(os.getenv("DOCMIND_MATCH_LOG_LIMIT", "60"))
    except Exception:
        max_match_log_lines = 60
    max_match_log_lines = max(0, min(max_match_log_lines, 500))
    match_log_count = 0
    suppressed_match_logs = 0
    body_term_hits: dict[str, set[int]] = {}

    def log_match_detail(msg: str):
        nonlocal match_log_count, suppressed_match_logs
        if match_log_count < max_match_log_lines:
            logger.debug(msg)
            match_log_count += 1
        else:
            suppressed_match_logs += 1

    now = datetime.datetime.now()
    for i in range(len(scores)):
        if i not in candidate_index_set:
            continue
        delta_days = (now - chunk_file_times[i]).days
        if delta_days < 30:
            time_weight = 1.0
        elif delta_days < 180:
            time_weight = 0.92
        elif delta_days < 365:
            time_weight = 0.85
        else:
            time_weight = 0.75
        scores[i] *= time_weight

    for i, chunk_text_item in enumerate(chunk_texts):
        if i not in candidate_index_set:
            continue
        path_no_ext = Path(chunk_paths[i]).stem.lower()
        chunk_text_lower = chunk_text_item.lower()

        for term in search_terms:
            term_lower = term.lower()

            # ---------- 文件名命中 ----------
            if should_score_filename_term(term_lower) and term_lower in path_no_ext:
                if term_lower in EXTENSION_TERMS:
                    scores[i] += 0.02
                elif len(term_lower) <= 2:
                    scores[i] += 0.05
                    log_match_detail(f"   [chunk文件名短词命中降权] '{term}' -> {chunk_paths[i]}")
                else:
                    scores[i] += 0.25
                    log_match_detail(f"   [chunk文件名命中] '{term}' -> {chunk_paths[i]}")
                continue

            # ---------- 正文命中 ----------
            if not should_score_body_term(term_lower):
                continue

            if term_lower in chunk_text_lower:
                if term_lower in EXTENSION_TERMS:
                    scores[i] += 0.02
                elif term_lower.isascii() and term_lower.replace('_', '').isalnum():
                    # 英文标识符保留较高权重，但不要太离谱
                    scores[i] += 0.18
                    log_match_detail(f"   [chunk英文命中] '{term}' -> {chunk_paths[i]}")
                else:
                    scores[i] += 0.12
                    log_match_detail(f"   [chunk正文命中] '{term}' -> {chunk_paths[i]}")
                if term_lower not in EXTENSION_TERMS:
                    body_term_hits.setdefault(term_lower, set()).add(i)

        if is_company_lookup:
            bonus = company_hint_bonus(
                chunk_text_item,
                candidate_df=company_candidate_df,
                total_docs=company_docs_total,
            )
            if bonus > 0:
                scores[i] += bonus
                log_match_detail(f"   [chunk公司线索加权] +{bonus:.2f} -> {chunk_paths[i]}")

    if suppressed_match_logs > 0:
        logger.debug(f"   [命中日志限流] 已省略 {suppressed_match_logs} 条命中明细")
    shift_keywords = ["其他", "别的", "所有", "全局", "抛开", "除了", "另外", "换个", "不说", "那"]
    ignored_file = None
    if any(k in question for k in shift_keywords) or len(question) < 4:
        ignored_file = current_focus_file
        current_focus_file = None
        if ignored_file:
            logger.info(f"   🔄 [焦点释放] 检测到话题转移，临时屏蔽: {ignored_file}")

    is_result_set_followup_query = ("已知文件如下" in search_query) or ("候选文件如下" in search_query)
    temp_query = question.lower() if is_result_set_followup_query else (question + " " + search_query).lower()
    sorted_indices = sorted(range(len(repo_state.paths)), key=lambda k: len(repo_state.paths[k]), reverse=True)
    for i in sorted_indices:
        full_name = repo_state.paths[i].lower()
        base_name = Path(repo_state.paths[i]).stem.lower()
        if ignored_file and full_name == ignored_file.lower():
            continue
        if full_name in temp_query:
            current_focus_file = repo_state.paths[i]
            logger.info(f"   🎯 [精确拦截-全名] -> {repo_state.paths[i]}")
            break
        import re
        pattern = rf"(?:^|[^a-zA-Z0-9_]){re.escape(base_name)}(?:[^a-zA-Z0-9_]|$)"
        if not base_name.isdigit() and re.search(pattern, temp_query):
            if temp_query.strip() == base_name or len(base_name) >= 4:
                current_focus_file = repo_state.paths[i]
                logger.info(f"   🎯 [精确拦截-词边界] -> {repo_state.paths[i]}")
                break

    if allowed_path_set is not None and current_focus_file and current_focus_file not in allowed_path_set:
        current_focus_file = None
    if current_focus_file and not any(k in question for k in shift_keywords):
        logger.info(f"   🔒 [全局焦点锁定] AI注意力集中于 -> {current_focus_file}")
        for i in range(len(scores)):
            if i in candidate_index_set and chunk_paths[i] == current_focus_file:
                scores[i] += 0.18

    is_macro_request = any(kw in question for kw in
                           ["时间线", "经过", "梳理", "复盘", "总结", "详细", "过程", "所有", "表现", "评价", "对吗",
                                "境遇", "怎么看", "经历", "待过"])
    is_compare_request = is_compare_intent_query(question)
    is_person_eval_query = (("评价" in question or "怎么看" in question or "这个人怎么样" in question) and any(
        len(term) >= 2 for term in search_terms))

    if is_person_eval_query:
        top_k, threshold = 6, 0.50
    elif is_entity_lookup and is_expansion_followup:
        top_k, threshold = 30, 0.28
        logger.info(f"   📚 [实体扩展增强] 上限扩至 {top_k} 份，及格线降至 {threshold}")
    elif is_entity_lookup:
        if is_company_lookup:
            top_k, threshold = 30, 0.30
            logger.info(f"   📚 [公司检索增强] 上限扩至 {top_k} 份，及格线降至 {threshold}")
        else:
            top_k, threshold = 20, 0.32
            logger.info(f"   📚 [实体检索增强] 上限扩至 {top_k} 份，及格线降至 {threshold}")
    elif is_compare_request:
        top_k, threshold = 24, 0.30
        logger.info(f"   📚 [比较题增强] 上限扩至 {top_k} 份，及格线降至 {threshold}")
    elif is_macro_request:
        top_k, threshold = 50, 0.28
        logger.info(f"   📂 [深度核查模式] 上限扩至 {top_k} 份，及格线降至 {threshold}")
    elif is_related_record_listing_query(question):
        top_k, threshold = 20, 0.33
        logger.info(f"   📚 [相关记录增强] 上限扩至 {top_k} 份，及格线降至 {threshold}")
    else:
        top_k, threshold = 12, 0.40

    ranked_candidate_indices = sorted(candidate_indices, key=lambda idx: float(scores[idx]), reverse=True)
    relevant_indices = [i for i in ranked_candidate_indices if scores[i] > threshold]
    weak_query = is_weak_query(question, search_terms)
    fallback_threshold = 0.24 if is_entity_lookup else 0.30

    if not relevant_indices and should_enable_fallback(search_terms, weak_query):
        relevant_indices = [i for i in ranked_candidate_indices if scores[i] > fallback_threshold]
        logger.info("   🛡️ [兜底打捞] 未达到阈值，启动底线捞取...")
    elif not relevant_indices and weak_query:
        logger.info("   🚫 [禁用兜底] 当前问题过弱或属于能力类问题，避免误捞上下文")

    if not relevant_indices and is_entity_lookup:
        relevant_indices = ranked_candidate_indices[:top_k] if ranked_candidate_indices else rescue_entity_lookup_indices(scores, top_k=top_k)
        logger.info(f"   🛟 [实体检索保底] 触发低阈值候选兜底，补入 {len(relevant_indices)} 个片段")

    is_file_lookup = is_file_location_lookup_query(question, search_query)
    if (is_file_lookup or is_compare_request) and body_term_hits:
        eligible_terms = []
        for term, hit_idx_set in body_term_hits.items():
            hit_count = len(hit_idx_set)
            if hit_count <= 0:
                continue
            # 过滤覆盖面过大的词，避免把“宽词”命中整体灌入上下文。
            if hit_count > 8:
                continue
            eligible_terms.append((term, hit_count))

        if eligible_terms:
            min_hit_count = min(hit_count for _, hit_count in eligible_terms)
            anchor_terms = [term for term, hit_count in eligible_terms if hit_count == min_hit_count]
            anchor_indices = []
            seen_anchor_idx = set()
            for term in anchor_terms:
                for idx in body_term_hits.get(term, set()):
                    if idx in seen_anchor_idx:
                        continue
                    seen_anchor_idx.add(idx)
                    anchor_indices.append(idx)

            anchor_indices.sort(key=lambda idx: float(scores[idx]), reverse=True)
            appended = 0
            for idx in anchor_indices:
                if idx in relevant_indices:
                    continue
                relevant_indices.append(idx)
                appended += 1

            if appended > 0 and is_file_lookup:
                logger.info(
                    f"   📍 [文件定位补召回] 基于命中词 {anchor_terms} 追加 {appended} 个片段"
                )

        if is_compare_request and len(relevant_indices) < 2:
            # 比较题至少补入一个“非当前首条”的正文命中片段，避免证据单源化。
            compare_candidates = []
            seen_compare_idx = set()
            for term, _ in eligible_terms:
                for idx in body_term_hits.get(term, set()):
                    if idx in seen_compare_idx or idx in relevant_indices:
                        continue
                    seen_compare_idx.add(idx)
                    compare_candidates.append(idx)

            compare_candidates.sort(key=lambda idx: float(scores[idx]), reverse=True)
            appended_compare = 0
            for idx in compare_candidates:
                relevant_indices.append(idx)
                appended_compare += 1
                if len(relevant_indices) >= 2 or appended_compare >= 2:
                    break

            if appended_compare > 0:
                logger.info(
                    f"   ⚖️ [比较题补证据] 追加 {appended_compare} 个正文命中片段用于对照"
                )

    relevant_indices = relevant_indices[:top_k]
    logger.info(f"   🔍 [溯源完毕] 本轮检索候选片段数量: {len(relevant_indices)}")

    return {"relevant_indices": relevant_indices, "scores": scores, "current_focus_file": current_focus_file, }
