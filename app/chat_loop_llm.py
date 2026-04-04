from __future__ import annotations

import os

import requests


def _call_local_ollama_answer(
    prompt: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
    *,
    timeout_sec: float,
    log_prefix: str,
) -> str | None:
    payload = {"model": ollama_model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(ollama_api_url, json=payload, timeout=timeout_sec)
        resp.raise_for_status()
        text = (resp.json().get("response") or "").strip()
        if not text:
            return None
        return " ".join(text.split())[:180]
    except Exception as e:
        logger.warning(f"⚠️ {log_prefix} 本地模型回答失败，回退固定文案: {e}")
        return None


def _is_smalltalk_local_llm_enabled() -> bool:
    raw = (os.getenv("DOCMIND_SMALLTALK_LOCAL_LLM") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _build_smalltalk_prompt(
    question: str,
    *,
    prev_user_question: str | None = None,
    prev_answer_preview: str | None = None,
    last_route: str | None = None,
) -> str:
    prev_q = (prev_user_question or "").strip() or "unknown"
    prev_a = (prev_answer_preview or "").strip() or "unknown"
    prev_route = (last_route or "").strip() or "unknown"
    return (
        "You are DocMind assistant. Reply in 1-2 concise Chinese sentences.\n"
        "Do not claim you viewed all local files. Keep it natural and direct.\n"
        f"Last route: {prev_route}\n"
        f"Last question: {prev_q}\n"
        f"Last answer preview: {prev_a}\n"
        f"User question: {question}\n"
        "Answer:"
    )


def _answer_smalltalk_with_local_llm(
    question: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
    *,
    prev_user_question: str | None = None,
    prev_answer_preview: str | None = None,
    last_route: str | None = None,
) -> str | None:
    if not _is_smalltalk_local_llm_enabled():
        return None
    prompt = _build_smalltalk_prompt(
        question,
        prev_user_question=prev_user_question,
        prev_answer_preview=prev_answer_preview,
        last_route=last_route,
    )
    return _call_local_ollama_answer(
        prompt=prompt,
        ollama_api_url=ollama_api_url,
        ollama_model=ollama_model,
        logger=logger,
        timeout_sec=20,
        log_prefix="smalltalk",
    )


def _is_out_of_scope_local_llm_enabled() -> bool:
    raw = (os.getenv("DOCMIND_OUT_OF_SCOPE_LOCAL_LLM") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def _build_out_of_scope_prompt(
    question: str,
    *,
    prev_user_question: str | None = None,
    prev_answer_preview: str | None = None,
    last_route: str | None = None,
) -> str:
    prev_q = (prev_user_question or "").strip() or "unknown"
    prev_a = (prev_answer_preview or "").strip() or "unknown"
    prev_route = (last_route or "").strip() or "unknown"
    return (
        "You are DocMind assistant. The question is out of document-retrieval scope.\n"
        "Reply in 1-2 concise Chinese sentences without claiming real-time data.\n"
        f"Last route: {prev_route}\n"
        f"Last question: {prev_q}\n"
        f"Last answer preview: {prev_a}\n"
        f"User question: {question}\n"
        "Answer:"
    )


def _answer_out_of_scope_with_local_llm(
    question: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
    *,
    prev_user_question: str | None = None,
    prev_answer_preview: str | None = None,
    last_route: str | None = None,
) -> str | None:
    if not _is_out_of_scope_local_llm_enabled():
        return None

    prompt = _build_out_of_scope_prompt(
        question,
        prev_user_question=prev_user_question,
        prev_answer_preview=prev_answer_preview,
        last_route=last_route,
    )
    return _call_local_ollama_answer(
        prompt=prompt,
        ollama_api_url=ollama_api_url,
        ollama_model=ollama_model,
        logger=logger,
        timeout_sec=20,
        log_prefix="out_of_scope",
    )
