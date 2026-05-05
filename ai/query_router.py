from __future__ import annotations

import json
import os
import requests

from ai.capability_smalltalk import answer_smalltalk
from ai.query_rewriter import is_local_smalltalk_intent, rewrite_search_query
from ai.query_router_rules import (
    _has_explicit_repo_meta_signal,
    _is_capability,
    _is_definitely_out_of_scope,
    _is_entity_lookup,
    _is_file_locator_query,
    _is_repo_meta,
    _is_stateful_smalltalk_followup,
    _normalize,
    _passes_inventory_route_guard,
    _should_preserve_contextual_retrieval,
    _should_try_local_inventory_route,
    _should_try_local_rewrite_for_smalltalk,
)


def _get_smalltalk_rewrite_timeout_sec() -> float:
    raw = (os.getenv("DOCMIND_SMALLTALK_REWRITE_TIMEOUT") or "").strip()
    if not raw:
        return 4.0

    try:
        value = float(raw)
    except ValueError:
        return 4.0

    return min(max(value, 0.5), 10.0)


def _is_smalltalk(question: str) -> bool:
    q = _normalize(question)
    if not q:
        return False

    if answer_smalltalk(question) is not None:
        return True

    return is_local_smalltalk_intent(question)


def _is_repo_meta_inventory_by_local_model(
    question: str,
    q: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
) -> bool:
    if not _should_try_local_inventory_route(question, q):
        return False

    prompt = f"""
你是一个中文问句分类器，只判断用户是不是在索要“知识库里的文档/文件清单”。
请只输出 JSON，不要解释。

判定为 true 的情况：
- 用户想知道当前有哪些文档/文件/资料
- 用户想看文档清单、文件列表、资料列表

判定为 false 的情况：
- 用户想在文档里查内容、实体、字段、结论
- 用户在问“文档里有哪些公司/人物/甲方/问题”
- 用户在定位某个内容在哪个文件

用户问题：
{question}

输出：
{{"inventory_listing": true}}
""".strip()

    try:
        resp = requests.post(
            ollama_api_url,
            json={
                "model": ollama_model,
                "prompt": prompt,
                "stream": False,
            },
            timeout=8,
        )
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        result = json.loads(text)
        is_inventory_listing = bool(result.get("inventory_listing"))
        if is_inventory_listing and _passes_inventory_route_guard(q):
            logger.info(f"🧭 [本地模型补判] inventory_listing -> {question}")
            return True
    except Exception as e:
        logger.warning(f"[inventory补判失败] {e}")

    return False


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
    state_hint = state_hint or {}
    last_route_hint = str(state_hint.get("last_route") or "").strip()
    last_user_question_hint = str(state_hint.get("last_user_question") or "").strip()
    last_answer_preview_hint = str(state_hint.get("last_answer_preview") or "").strip()
    last_effective_search_query_hint = str(state_hint.get("last_effective_search_query") or "").strip()

    if _is_capability(q):
        logger.info(f"🧭 [规则命中] capability -> {question}")
        return {"route": "system_capability"}

    if is_rule_smalltalk:
        logger.info(f"🧭 [规则命中] smalltalk -> {question}")
        return {"route": "smalltalk"}

    if _is_repo_meta_inventory_by_local_model(question, q, ollama_api_url, ollama_model, logger):
        return {"route": "repo_meta"}

    if _is_file_locator_query(q):
        logger.info(f"🧭 [规则命中] file_locator -> {question}")
        return {"route": "normal_retrieval"}

    if _is_entity_lookup(q):
        logger.info(f"🧭 [规则命中] entity_lookup -> {question}")
        return {"route": "normal_retrieval"}

    if _is_repo_meta(q):
        logger.info(f"🧭 [规则命中] repo_meta -> {question}")
        return {"route": "repo_meta"}

    try:
        from app.dialog.repo_meta_rules import is_repo_meta_request

        if is_repo_meta_request(question):
            logger.info(f"🧭 [规则命中] repo_meta(local) -> {question}")
            return {"route": "repo_meta"}
    except Exception:
        pass

    if _should_preserve_contextual_retrieval(
        question,
        q,
        state_hint,
        is_rule_smalltalk=is_rule_smalltalk,
    ):
        logger.info(f"🧭 [规则命中] contextual_followup -> {question}")
        return {"route": "normal_retrieval"}

    if _is_definitely_out_of_scope(q):
        logger.info(f"🧭 [规则命中] out_of_scope -> {question}")
        return {"route": "out_of_scope"}

    if _is_smalltalk_by_local_rewrite(question, ollama_api_url, ollama_model, logger):
        logger.info(f"🧭 [本地引擎路由补判] smalltalk -> {question}")
        return {"route": "smalltalk"}

    prompt = f"""
你是一个问句路由器，只负责判断用户问题属于哪一类。
请只输出 JSON，不要输出解释。

对话上下文：
- 上一轮路由: {last_route_hint or "unknown"}
- 上一轮用户问题: {last_user_question_hint or "unknown"}
- 上一轮回答摘要: {last_answer_preview_hint or "unknown"}
- 上一轮有效检索词: {last_effective_search_query_hint or "unknown"}

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
        if route == "out_of_scope" and (
            _should_preserve_contextual_retrieval(
                question,
                q,
                state_hint,
                is_rule_smalltalk=is_rule_smalltalk,
            )
            or not _is_definitely_out_of_scope(q)
        ):
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
