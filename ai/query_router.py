from __future__ import annotations

import json
import re
import requests


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


def _is_smalltalk(q: str) -> bool:
    patterns = ["你好", "谢谢", "哈哈", "好的", "行", "在吗"]
    return any(p in q for p in patterns) and len(q) <= 8


def _is_repo_meta(q: str) -> bool:
    patterns = ["多少文件", "多少个文件", "文件数量", "有哪些格式", "文件格式"]
    return any(p in q for p in patterns)


def route_question(question: str, ollama_api_url: str, ollama_model: str, logger) -> dict:
    q = _normalize(question)

    # ✅ 第一层：极少量护栏（避免严重误判）
    if _is_capability(q):
        logger.info(f"🧭 [规则命中] capability -> {question}")
        return {"route": "system_capability"}

    if _is_smalltalk(q):
        logger.info(f"🧭 [规则命中] smalltalk -> {question}")
        return {"route": "smalltalk"}

    if _is_repo_meta(q):
        logger.info(f"🧭 [规则命中] repo_meta -> {question}")
        return {"route": "repo_meta"}

    # ✅ 第二层：交给 AI 判断（主体逻辑）
    prompt = f"""
你是一个问句路由器，只负责判断用户问题属于哪一类。
请只输出 JSON，不要输出解释。

分类：
- system_capability
- repo_meta
- inventory
- smalltalk
- normal_retrieval

用户问题：
{question}

输出：
{{"route": "..."}}
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

        if route not in {
            "system_capability",
            "repo_meta",
            "inventory",
            "smalltalk",
            "normal_retrieval",
        }:
            route = "normal_retrieval"

        return {"route": route}

    except Exception as e:
        logger.warning(f"[路由失败] {e}")
        return {"route": "normal_retrieval"}