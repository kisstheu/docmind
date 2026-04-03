from __future__ import annotations

import json
import os
import re
import requests

from ai.capability_smalltalk import answer_smalltalk
from ai.query_rewriter import is_local_smalltalk_intent, rewrite_search_query


def _get_smalltalk_rewrite_timeout_sec() -> float:
    raw = (os.getenv("DOCMIND_SMALLTALK_REWRITE_TIMEOUT") or "").strip()
    if not raw:
        return 2.5

    try:
        value = float(raw)
    except ValueError:
        return 2.5

    return min(max(value, 0.5), 10.0)


def _normalize(q: str) -> str:
    q = q.strip().lower()
    q = re.sub(r"[？?！!，,。\.、\s]+", "", q)
    return q


def _is_capability(q: str) -> bool:
    patterns = [
        "你是谁", "你是啥",
        "你能做", "你可以做",
        "能做啥", "干啥", "做啥",
        "怎么用", "功能", "help",
    ]
    return any(p in q for p in patterns)


def _is_smalltalk(question: str) -> bool:
    q = _normalize(question)
    if not q:
        return False

    if answer_smalltalk(question) is not None:
        return True

    return is_local_smalltalk_intent(question)


def _is_repo_meta(q: str) -> bool:
    if _is_file_locator_query(q):
        return False

    has_name_content_mismatch = _is_name_content_mismatch(q)
    has_list_doc_query = _is_list_doc_query(q)
    has_doc_word = any(x in q for x in ["文件", "文档", "资料"])
    has_meta_word = any(x in q for x in [
        "多少", "数量", "格式", "分类", "清单",
        "最新", "最早", "最晚", "最近更新", "最近修改",
        "修改时间", "创建时间", "占多大", "总大小", "空间",
    ])
    return has_name_content_mismatch or has_list_doc_query or (has_doc_word and has_meta_word)


def _has_explicit_repo_meta_signal(question: str, q: str) -> bool:
    if _is_repo_meta(q):
        return True
    try:
        from app.dialog.repo_meta_rules import is_repo_meta_request

        return bool(is_repo_meta_request(question))
    except Exception:
        return False


def _is_file_locator_query(q: str) -> bool:
    merged = re.sub(r"\s+", "", (q or "").lower())
    if not merged:
        return False

    direct_patterns = [
        "在哪个文件", "在那个文件", "是哪个文件", "是那个文件",
        "哪个文件", "哪些文件", "哪份文件", "文件里", "文件中",
        "在哪个文档", "在那个文档", "是哪个文档", "是那个文档",
        "哪个文档", "哪些文档",
        "在哪个记录", "是哪个记录", "哪个记录", "哪些记录",
    ]
    return any(p in merged for p in direct_patterns)


def _is_entity_lookup(q: str) -> bool:
    has_doc_word = any(x in q for x in ["文件", "文档", "资料"])
    if has_doc_word:
        return False

    has_entity_word = any(x in q for x in [
        "公司名", "公司名称", "公司", "企业",
        "人名", "姓名", "人物",
        "项目名", "项目名称", "项目",
    ])
    has_lookup_intent = any(x in q for x in [
        "找", "查", "搜",
        "提到", "提及",
        "名字", "名称",
        "哪些", "哪个", "哪位", "哪几个", "哪几家", "列出",
        "对应", "分别", "关联", "匹配",
    ])
    has_repo_meta_intent = any(x in q for x in [
        "多少", "数量", "格式", "分类", "清单",
        "最新", "最早", "最晚", "最近更新", "最近修改",
        "修改时间", "创建时间", "占多大", "总大小", "空间",
    ])

    return has_entity_word and has_lookup_intent and not has_repo_meta_intent


def _is_name_content_mismatch(q: str) -> bool:
    has_name_word = any(x in q for x in ["文件名", "标题", "题目", "名称", "名字"])
    has_content_word = any(x in q for x in ["内容", "正文"])
    has_mismatch_word = any(x in q for x in ["不符", "不一致", "不匹配", "对不上", "冲突", "矛盾"])
    return has_name_word and has_content_word and has_mismatch_word


def _is_list_doc_query(q: str) -> bool:
    has_doc_word = any(x in q for x in ["文件", "文档", "资料"])
    has_list_word = any(x in q for x in ["列出", "列一下", "列下", "列出来", "清单", "罗列", "展开"])
    return has_doc_word and has_list_word


def _is_definitely_out_of_scope(q: str) -> bool:
    if not q:
        return False

    in_scope_markers = (
        "文件", "文档", "资料", "笔记", "记录", "截图",
        "仓库", "目录", "索引", "缓存",
        "公司", "企业", "项目", "人物", "人名", "姓名",
        "内容", "正文", "标题", "文件名",
        "提到", "提及", "出现", "在哪", "位置",
        "整理", "梳理", "总结", "归纳", "分析", "统计", "分类",
        "清单", "列出", "查", "找", "搜", "检索",
        "最近", "最早", "最晚", "时间线", "多少", "数量",
        "格式", "创建时间", "修改时间",
    )
    if any(t in q for t in in_scope_markers):
        return False

    assistant_target_markers = ("你", "你们", "助手", "机器人", "docmind", "ai")
    if not any(t in q for t in assistant_target_markers):
        return False

    external_realtime_markers = (
        "天气", "气温", "温度", "下雨", "空气质量",
        "股价", "汇率", "油价", "新闻", "热搜", "比分", "彩票",
    )
    if any(t in q for t in external_realtime_markers):
        return True

    # 面向助手且不含文档意图的泛问题，若本地也不判为闲聊，则视为越界。
    return not is_local_smalltalk_intent(q)


def _should_try_local_rewrite_for_smalltalk(q: str) -> bool:
    if not q or len(q) > 16:
        return False

    block_terms = (
        "文件",
        "文档",
        "资料",
        "公司",
        "项目",
        "他",
        "她",
        "对方",
        "关于什么",
        "什么内容",
        "什么主题",
        "主题",
        "内容",
    )
    if any(t in q for t in block_terms):
        return False

    return True


def _is_smalltalk_by_local_rewrite(question: str, ollama_api_url: str, ollama_model: str, logger) -> bool:
    q = _normalize(question)
    if not _should_try_local_rewrite_for_smalltalk(q):
        return False

    timeout_sec = _get_smalltalk_rewrite_timeout_sec()
    rewritten = rewrite_search_query(
        question,
        [],
        ollama_api_url,
        ollama_model,
        logger,
        timeout_sec=timeout_sec,
        silent_fail=True,
    )
    rewritten_norm = _normalize(rewritten)
    if not rewritten_norm or rewritten_norm == q:
        return False

    if any(t in rewritten_norm for t in ("文件", "文档", "资料", "公司", "项目", "记录")):
        return False

    return is_local_smalltalk_intent(rewritten_norm)


def _is_stateful_smalltalk_followup(q: str, state_hint: dict | None) -> bool:
    if not state_hint:
        return False

    last_route = _normalize(str(state_hint.get("last_route") or ""))
    if last_route != "smalltalk":
        return False

    if not q or len(q) > 12:
        return False

    retrieval_markers = (
        "找", "查", "搜", "检索",
        "文件", "文档", "资料", "记录",
        "公司", "人物", "人名", "项目",
        "时间", "日期", "最近", "最早", "最晚",
        "多少", "哪些", "哪几个", "哪几家",
        "列出", "清单", "提到", "提及", "在哪", "位置",
        "帮我", "麻烦", "请你",
    )
    if any(t in q for t in retrieval_markers):
        return False

    return True


def route_question(
    question: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
    *,
    state_hint: dict | None = None,
) -> dict:
    q = _normalize(question)
    is_rule_smalltalk = _is_smalltalk(question)

    # ✅ 第一层：极少量护栏（避免严重误判）
    if _is_capability(q):
        logger.info(f"🧭 [规则命中] capability -> {question}")
        return {"route": "system_capability"}

    if is_rule_smalltalk:
        logger.info(f"🧭 [规则命中] smalltalk -> {question}")
        return {"route": "smalltalk"}

    if _is_file_locator_query(q):
        logger.info(f"🧭 [规则命中] file_locator -> {question}")
        return {"route": "normal_retrieval"}

    if _is_entity_lookup(q):
        logger.info(f"🧭 [规则命中] entity_lookup -> {question}")
        return {"route": "normal_retrieval"}

    if _is_repo_meta(q):
        logger.info(f"🧭 [规则命中] repo_meta -> {question}")
        return {"route": "repo_meta"}

    if _is_definitely_out_of_scope(q):
        logger.info(f"🧭 [规则命中] out_of_scope -> {question}")
        return {"route": "out_of_scope"}

    if _is_smalltalk_by_local_rewrite(question, ollama_api_url, ollama_model, logger):
        logger.info(f"🧭 [本地引擎路由补判] smalltalk -> {question}")
        return {"route": "smalltalk"}

    state_hint = state_hint or {}
    last_route_hint = str(state_hint.get("last_route") or "").strip()
    last_user_question_hint = str(state_hint.get("last_user_question") or "").strip()
    last_answer_preview_hint = str(state_hint.get("last_answer_preview") or "").strip()

    prompt = f"""
你是一个问句路由器，只负责判断用户问题属于哪一类。
请只输出 JSON，不要输出解释。

对话上下文：
- 上一轮路由: {last_route_hint or "unknown"}
- 上一轮用户问题: {last_user_question_hint or "unknown"}
- 上一轮回答摘要: {last_answer_preview_hint or "unknown"}

分类：
- system_capability
- repo_meta
- inventory
- smalltalk
- out_of_scope
- normal_retrieval

补充规则：
1) 若上一轮路由是 smalltalk，且当前问句是短句/残句/承接语，并且不包含明确文档检索意图，优先 smalltalk。
2) 涉及文件/文档/记录/公司/项目的检索、统计、定位问题，不要判 smalltalk。
3) 当 route=smalltalk 时，请顺带生成一条可直接回复用户的简短中文（1-2句），放在 smalltalk_reply 字段。
4) 当 route 不是 smalltalk 时，smalltalk_reply 置空字符串。

用户问题：
{question}

输出：
{{"route": "...", "smalltalk_reply": ""}}
""".strip()

    try:
        resp = requests.post(
            ollama_api_url,
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=20,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        result = json.loads(text)

        route = result.get("route", "normal_retrieval")
        smalltalk_reply = (result.get("smalltalk_reply") or "").strip()

        if route not in {
            "system_capability",
            "repo_meta",
            "inventory",
            "smalltalk",
            "out_of_scope",
            "normal_retrieval",
        }:
            route = "normal_retrieval"

        # 模型偶发会把正常问题误判为 smalltalk，做一次保守收敛。
        if route == "smalltalk" and not (is_rule_smalltalk or _is_stateful_smalltalk_followup(q, state_hint)):
            route = "normal_retrieval"
        if route == "out_of_scope" and not _is_definitely_out_of_scope(q):
            route = "normal_retrieval"
        if route == "system_capability" and not _is_capability(q):
            route = "normal_retrieval"
        if route == "repo_meta" and not _has_explicit_repo_meta_signal(question, q):
            route = "normal_retrieval"

        if route != "smalltalk":
            smalltalk_reply = ""

        return {"route": route, "smalltalk_reply": smalltalk_reply[:180]}

    except Exception as e:
        logger.warning(f"[路由失败] {e}")
        return {"route": "normal_retrieval"}
