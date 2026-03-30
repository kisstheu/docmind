from __future__ import annotations

import datetime
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from retrieval.chunking import expand_neighbor_chunks
from retrieval.query_utils import (EXTENSION_TERMS, classify_org_candidate, detect_inventory_target,
                                   extract_company_candidates, extract_query_terms, )


def is_capability_like_query(question: str) -> bool:
    q = question.strip().lower()
    patterns = ["你是谁", "你能做什么", "你可以做什么", "能做什么", "可以做什么", "能干啥", "干啥", "做啥", "能做啥",
        "有什么功能", "有啥功能", "你的功能", "怎么用", "介绍一下", "help", ]
    return any(p in q for p in patterns)


def is_relation_mismatch_query(question: str) -> bool:
    q = (question or "").replace(" ", "")
    patterns = ["不符", "不一致", "不匹配", "对不上", "冲突", "矛盾", "是否一致", "有没有一致", "文件名", "标题", "正文"]
    return any(p in q for p in patterns)


def is_entity_lookup_query(question: str) -> bool:
    q = (question or "").replace(" ", "").lower()
    patterns = [
        "公司名", "公司名称", "企业名称", "单位名称", "组织名称",
        "人名", "姓名", "名字",
        "项目名", "项目名称",
    ]
    return any(p in q for p in patterns)


def is_expansion_followup_question(question: str) -> bool:
    q = (question or "").replace(" ", "").lower()
    if not q:
        return False
    patterns = ["更多", "还有", "另外", "其他", "别的", "继续", "再来", "补充", "还包括"]
    return any(p in q for p in patterns)


def is_entity_name_lookup_context(question: str, search_terms: list[str]) -> bool:
    if is_entity_lookup_query(question):
        return True

    term_set = {t.strip().lower() for t in (search_terms or []) if t and t.strip()}
    if "名称" in term_set:
        if term_set.intersection({"公司", "人名", "姓名", "项目"}):
            return True
    if term_set.intersection({"公司名", "公司名称", "企业名称", "单位名称", "组织名称", "项目名称"}):
        return True

    return False


def is_company_name_lookup_context(question: str, search_terms: list[str]) -> bool:
    q = (question or "").replace(" ", "").lower()
    if any(x in q for x in ["公司名", "公司名称", "企业名称", "单位名称", "组织名称"]):
        return True

    term_set = {t.strip().lower() for t in (search_terms or []) if t and t.strip()}
    if term_set.intersection({"公司", "公司名", "公司名称", "企业", "企业名称", "单位", "单位名称", "组织", "组织名称"}):
        return True
    if "名称" in term_set and term_set.intersection({"公司", "企业", "单位", "组织"}):
        return True
    return False


def is_weak_query(question: str, search_terms: list[str]) -> bool:
    q = question.strip().lower()

    if is_capability_like_query(q):
        return True

    # “查实体名称”类是明确检索需求，不按弱问题处理。
    if is_entity_lookup_query(q):
        return False

    if not search_terms:
        return True

    joined = "".join(search_terms).strip()
    if len(search_terms) == 1 and len(joined) <= 2:
        return True

    weak_terms = {"能做啥", "做啥", "干啥", "功能", "帮助", "help"}
    if any(term in weak_terms for term in search_terms):
        return True

    return False


def should_enable_fallback(search_terms: list[str], weak_query: bool) -> bool:
    if weak_query:
        return False
    return bool(search_terms)


def rescue_entity_lookup_indices(scores: np.ndarray, top_k: int) -> list[int]:
    if scores.size == 0 or top_k <= 0:
        return []

    sorted_idx = [int(i) for i in np.argsort(scores)[::-1]]
    strong_hits = [i for i in sorted_idx if float(scores[i]) > 0.18][:top_k]
    if strong_hits:
        return strong_hits

    return sorted_idx[: min(top_k, 8)]


def determine_query_flags(question: str):
    inventory_triggers = ["多少", "哪些", "有哪些", "提到", "涉及", "所有", "盘点", "过"]
    relationship_queries = ["对我如何", "关系好", "评价", "他人怎么样", "对他"]
    greetings = ["你好", "嗨", "在吗", "谢谢", "好的", "ok", "嗯", "哈哈", "知道了", "原来如此", "厉害", "棒", "牛逼",
                 "多谢", "感谢"]
    system_queries = ["能干啥", "你是谁", "怎么用", "你能做什么", "你的功能", "介绍一下", "能做什么", "可以做什么",
        "你可以做什么", "有什么功能", "有啥功能"]

    inventory_target_type, inventory_target_label = detect_inventory_target(question)
    is_inventory_query = inventory_target_type is not None and any(t in question for t in inventory_triggers)
    is_relationship_query = any(q in question for q in relationship_queries)
    skip_retrieval = False
    q_lower = question.lower().strip()
    # 移除标点符号后的纯文本长度
    import re
    pure_text = re.sub(r'[^\w\s]', '', q_lower)

    # 只有当用户输入的内容基本上全是问候语（比如长度极短），或者是精确等于问候语时，才跳过
    is_pure_greeting = any(q_lower == g or pure_text == g for g in greetings)

    if is_pure_greeting and not is_relationship_query:
        skip_retrieval = True
    elif any(word in question for word in system_queries):
        skip_retrieval = True

    return {"inventory_target_type": inventory_target_type, "inventory_target_label": inventory_target_label,
        "is_inventory_query": is_inventory_query, "is_relationship_query": is_relationship_query,
        "skip_retrieval": skip_retrieval, }


def is_over_generic_term(term: str) -> bool:
    t = (term or "").strip().lower()

    generic_terms = {"公司", "事情", "情况", "问题", "内容", "资料", "文件", "记录", "信息", "人员", "地方", "时间",
        "东西", "方面", }

    return t in generic_terms


def is_date_like_term(term: str) -> bool:
    t = (term or "").strip().lower()

    # 纯数字，且位数很短，通常是日期/编号噪音
    if re.fullmatch(r"\d{1,4}", t):
        return True

    # 常见中文日期表达
    if re.fullmatch(r"\d{1,2}号", t):
        return True
    if re.fullmatch(r"\d{1,2}日", t):
        return True
    if re.fullmatch(r"\d{1,2}月", t):
        return True
    if re.fullmatch(r"\d{4}年", t):
        return True
    if re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", t):
        return True

    return False


def is_sentence_like_term(term: str) -> bool:
    t = (term or "").strip()
    if not t:
        return False
    if term in {"动作", "情况", "事情"}:
        return True
    # 太长的中文串，大概率不是关键词，而是整句/短语
    if re.fullmatch(r"[\u4e00-\u9fa5]{7,}", t):
        return True

    # 明显带动作语气的短句
    if re.search(r"(给我|帮我|我想|我先|分析|整理|梳理|看下|看看|说说|讲讲)", t):
        return True

    return False


def is_result_set_boilerplate_term(term: str) -> bool:
    t = (term or "").strip()
    if not t:
        return False
    patterns = [
        "已知文件如下",
        "已知文档如下",
        "候选文件如下",
        "候选文档如下",
        "请基于这些文件的内容回答",
        "请基于这些文档的内容回答",
    ]
    return any(p in t for p in patterns)


def should_score_filename_term(term: str) -> bool:
    t = (term or "").strip().lower()
    if not t:
        return False
    if t in {"屏幕", "截图", "屏幕截图", "图片", "图像"}:
        return False
    if is_date_like_term(t):
        return False
    if is_sentence_like_term(t):
        return False
    return True


def build_company_candidate_df(docs: list[str]) -> Tuple[Dict[str, int], int]:
    candidate_df: Dict[str, int] = {}
    total_docs = 0

    for doc_text in docs or []:
        total_docs += 1
        seen_in_doc = set()
        for name in extract_company_candidates(doc_text or ""):
            if classify_org_candidate(name) == "generic":
                continue
            seen_in_doc.add(name)

        for name in seen_in_doc:
            candidate_df[name] = candidate_df.get(name, 0) + 1

    return candidate_df, max(1, total_docs)


def company_hint_bonus(
    chunk_text: str,
    candidate_df: dict[str, int] | None = None,
    total_docs: int = 1,
) -> float:
    text = chunk_text or ""
    if not text:
        return 0.0

    org_hits = extract_company_candidates(text)
    if not org_hits:
        return 0.0

    quality_sum = 0.0
    total_docs = max(1, int(total_docs or 1))
    denom = float(np.log(total_docs + 1.0))
    if denom <= 0:
        denom = 1.0

    title_role_re = r"(?:高级|资深|首席|招聘|HR|人事|经理|总监|负责人|创始人|CEO|CTO|COO|CFO|VP)"
    for name in org_hits:
        kind = classify_org_candidate(name)
        if kind == "generic":
            continue

        quality = 1.0 if kind == "explicit" else 0.75
        if re.search(rf"{re.escape(name)}\s*[·•]\s*{title_role_re}", text, flags=re.IGNORECASE):
            quality += 0.20

        df = 1
        if candidate_df is not None:
            df = max(1, int(candidate_df.get(name, 1)))

        idf = float(np.log((total_docs + 1.0) / (df + 1.0)) / denom)
        idf = max(0.15, min(1.0, idf))
        quality_sum += quality * idf

    if quality_sum <= 0:
        return 0.0

    return min(0.32, 0.20 * quality_sum)


def should_score_body_term(term: str) -> bool:
    t = (term or "").strip().lower()
    if not t:
        return False

    if is_over_generic_term(t):
        return False
    if is_date_like_term(t):
        return False
    if is_sentence_like_term(t):
        return False

    # 中文词太短，默认不参与正文打分
    if re.fullmatch(r"[\u4e00-\u9fa5]+", t) and len(t) <= 2:
        return False

    # 纯英文/数字太短，也不参与正文打分
    if re.fullmatch(r"[a-z0-9_]+", t) and len(t) <= 2:
        return False

    return True


def is_related_record_listing_query(question: str) -> bool:
    q = (question or "").replace(" ", "").lower()
    has_related = any(x in q for x in ["有关的记录", "相关记录", "有关记录", "相关的记录", "有关文档", "相关文档"])
    has_listing = any(x in q for x in ["哪些", "哪几", "有哪", "最近"])
    return has_related and has_listing


def perform_retrieval(question: str, search_query: str, repo_state, model_emb, logger, current_focus_file: str | None,
        context_anchor: str = "", ):
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

    def log_match_detail(msg: str):
        nonlocal match_log_count, suppressed_match_logs
        if match_log_count < max_match_log_lines:
            logger.debug(msg)
            match_log_count += 1
        else:
            suppressed_match_logs += 1

    now = datetime.datetime.now()
    for i in range(len(scores)):
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

    if current_focus_file and not any(k in question for k in shift_keywords):
        logger.info(f"   🔒 [全局焦点锁定] AI注意力集中于 -> {current_focus_file}")
        for i in range(len(scores)):
            if chunk_paths[i] == current_focus_file:
                scores[i] += 0.18

    is_macro_request = any(kw in question for kw in
                           ["时间线", "经过", "梳理", "复盘", "总结", "详细", "过程", "所有", "表现", "评价", "对吗",
                                "境遇", "怎么看", "经历", "待过"])
    is_compare_request = any(kw in question for kw in ["不同", "区别", "差异", "异同", "比较", "对比", "相同", "一样", "一致"])
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

    relevant_indices = [i for i in np.argsort(scores)[::-1] if scores[i] > threshold]
    weak_query = is_weak_query(question, search_terms)
    fallback_threshold = 0.24 if is_entity_lookup else 0.30

    if not relevant_indices and should_enable_fallback(search_terms, weak_query):
        relevant_indices = [i for i in np.argsort(scores)[::-1] if scores[i] > fallback_threshold]
        logger.info("   🛡️ [兜底打捞] 未达到阈值，启动底线捞取...")
    elif not relevant_indices and weak_query:
        logger.info("   🚫 [禁用兜底] 当前问题过弱或属于能力类问题，避免误捞上下文")

    if not relevant_indices and is_entity_lookup:
        relevant_indices = rescue_entity_lookup_indices(scores, top_k=top_k)
        logger.info(f"   🛟 [实体检索保底] 触发低阈值候选兜底，补入 {len(relevant_indices)} 个片段")

    relevant_indices = relevant_indices[:top_k]
    logger.info(f"   🔍 [溯源完毕] 最终喂给大模型的片段数量: {len(relevant_indices)}")

    return {"relevant_indices": relevant_indices, "scores": scores, "current_focus_file": current_focus_file, }


def build_context_text(relevant_indices: List[int], repo_state, logger) -> str:
    if not relevant_indices:
        return ""

    expanded_indices = expand_neighbor_chunks(
        top_chunk_indices=relevant_indices,
        chunk_paths=repo_state.chunk_paths,
        chunk_meta=repo_state.chunk_meta,
        neighbor=1,
    )

    file_chunk_count = {}
    filtered_indices = []
    for idx in expanded_indices:
        p = repo_state.chunk_paths[idx]
        file_chunk_count.setdefault(p, 0)
        if file_chunk_count[p] >= 3:
            continue
        filtered_indices.append(idx)
        file_chunk_count[p] += 1

    context_blocks = []
    for idx in filtered_indices:
        meta = repo_state.chunk_meta[idx]
        context_blocks.append(
            f"文件【{repo_state.chunk_paths[idx]}】（chunk #{meta['chunk_id']}，位置 {meta['start']}-{meta['end']}）：\n{repo_state.chunk_texts[idx]}"
        )

    logger.debug(
        f"本次最终送入大模型的chunk文件列表: {list(dict.fromkeys([repo_state.chunk_paths[idx] for idx in filtered_indices]))}"
    )
    return "【参考片段】:\n" + "\n---\n".join(context_blocks) + "\n\n"


def build_inventory_candidates_text(question: str, repo_state, inventory_target_type: str | None) -> str:
    if inventory_target_type != "company":
        return ""

    candidate_pool = []
    for doc_text in repo_state.docs:
        candidate_pool.extend(extract_company_candidates(doc_text))

    unique_names = []
    for name in candidate_pool:
        if name not in unique_names:
            unique_names.append(name)

    deduped_names = []
    for name in unique_names:
        if any(name != other and name in other and len(other) >= len(name) + 2 for other in unique_names):
            continue
        deduped_names.append(name)

    explicit_names, ambiguous_names, generic_names = [], [], []
    for name in deduped_names:
        kind = classify_org_candidate(name)
        if kind == "explicit":
            explicit_names.append(name)
        elif kind == "ambiguous":
            ambiguous_names.append(name)
        else:
            generic_names.append(name)

    def uniq_keep_order(items):
        result = []
        for x in items:
            if x not in result:
                result.append(x)
        return result

    explicit_names = uniq_keep_order(explicit_names)
    ambiguous_names = uniq_keep_order(ambiguous_names)
    generic_names = uniq_keep_order(generic_names)

    lines = []
    if explicit_names or ambiguous_names or generic_names:
        lines.append("【盘点候选：组织】")
        if explicit_names:
            lines.append("【明确组织名】")
            lines.extend([f"- {x}" for x in explicit_names[:20]])
        if ambiguous_names:
            lines.append("【可能是简称或未写全】")
            lines.extend([f"- {x}" for x in ambiguous_names[:20]])
        if generic_names:
            lines.append("【泛称/指代】")
            lines.extend([f"- {x}" for x in generic_names[:20]])

    return "\n".join(lines) + "\n\n" if lines else ""
