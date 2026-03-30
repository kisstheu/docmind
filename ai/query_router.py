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
    has_name_content_mismatch = _is_name_content_mismatch(q)
    has_list_doc_query = _is_list_doc_query(q)
    has_doc_word = any(x in q for x in ["文件", "文档", "资料"])
    has_meta_word = any(x in q for x in [
        "多少", "数量", "格式", "分类", "清单",
        "最新", "最早", "最晚", "最近更新", "最近修改",
        "修改时间", "创建时间", "占多大", "总大小", "空间",
    ])
    return has_name_content_mismatch or has_list_doc_query or (has_doc_word and has_meta_word)


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
        "哪些", "哪几个", "哪几家", "列出",
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


def route_question(question: str, ollama_api_url: str, ollama_model: str, logger) -> dict:
    q = _normalize(question)

    # ✅ 第一层：极少量护栏（避免严重误判）
    if _is_capability(q):
        logger.info(f"🧭 [规则命中] capability -> {question}")
        return {"route": "system_capability"}

    if _is_smalltalk(q):
        logger.info(f"🧭 [规则命中] smalltalk -> {question}")
        return {"route": "smalltalk"}

    if _is_entity_lookup(q):
        logger.info(f"🧭 [规则命中] entity_lookup -> {question}")
        return {"route": "normal_retrieval"}

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

        # 模型偶发会把正常问题误判为 smalltalk，做一次保守收敛。
        if route == "smalltalk" and not _is_smalltalk(q):
            route = "normal_retrieval"

        return {"route": route}

    except Exception as e:
        logger.warning(f"[路由失败] {e}")
        return {"route": "normal_retrieval"}
