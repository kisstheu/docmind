from __future__ import annotations

import requests


def _call_local_ollama(prompt: str, logger, ollama_api_url: str, ollama_model: str) -> str:
    logger.info("   🤖 正在使用本地模型做大方面概括...")
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "stream": False,
    }
    response = requests.post(ollama_api_url, json=payload, timeout=180)
    response.raise_for_status()
    text = response.json().get("response", "").strip()
    logger.info(f"      ✨ 本地模型概括输出：[{text[:120]}]")
    return text


def build_topic_summarizer(logger, ollama_api_url: str, ollama_model: str):
    def _summarizer(prompt: str) -> str:
        return _call_local_ollama(
            prompt,
            logger=logger,
            ollama_api_url=ollama_api_url,
            ollama_model=ollama_model,
        )

    return _summarizer


def resolve_route(question: str, event, ollama_api_url: str, ollama_model: str, logger, state=None) -> dict:
    if event.route_hint:
        route = event.route_hint
        logger.info(f"🧭 [状态机事件] {event.name}: {question} -> {route}")
        return {"route": route, "smalltalk_reply": "", "route_question_input": question}

    from ai.query_router import route_question

    state_hint = None
    if state is not None:
        state_hint = {
            "last_route": getattr(state, "last_route", None),
            "last_user_question": getattr(state, "last_user_question", None),
            "last_answer_preview": getattr(state, "last_answer_preview", None),
            "last_effective_search_query": getattr(state, "last_effective_search_query", None),
        }

    route_question_input = question
    if getattr(event, "name", "") == "query_correction" and getattr(event, "merged_query", None):
        route_question_input = event.merged_query

    route_info = route_question(
        route_question_input,
        ollama_api_url,
        ollama_model,
        logger,
        state_hint=state_hint,
    )
    route = route_info.get("route", "normal_retrieval")
    route_info["route"] = route
    route_info.setdefault("smalltalk_reply", "")
    route_info.setdefault("route_question_input", route_question_input)
    logger.info(f"🧭 [模型路由] 问题: {question} -> {route}")
    return route_info
