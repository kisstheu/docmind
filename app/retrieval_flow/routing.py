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


def resolve_route(question: str, event, ollama_api_url: str, ollama_model: str, logger) -> str:
    if event.route_hint:
        route = event.route_hint
        logger.info(f"🧭 [状态机事件] {event.name}: {question} -> {route}")
        return route

    from ai.query_router import route_question

    route_info = route_question(question, ollama_api_url, ollama_model, logger)
    route = route_info["route"]
    logger.info(f"🧭 [模型路由] 问题: {question} -> {route}")
    return route
