from __future__ import annotations

import re
import time

import requests
from google.genai import types

from ai.capabilities import (
    answer_repo_meta_question,
    answer_smalltalk,
    answer_system_capability_question,
)
from ai.prompt_builder import build_final_prompt
from ai.query_rewriter import rewrite_search_query
from app.dialog_state_machine import (
    ConversationState,
    apply_event_to_state,
    detect_dialog_event,
)
from retrieval.search_engine import (
    build_context_text,
    build_inventory_candidates_text,
    determine_query_flags,
    perform_retrieval,
)


def normalize_colloquial_question(question: str) -> str:
    q = question.strip()

    replacements = [
        (r"找个?仁儿", "找人"),
        (r"找个?仁", "找人"),
        (r"找个?银", "找人"),
        (r"找个?人儿", "找人"),
        (r"仁儿", "人"),
        (r"\b仁\b", "人"),
        (r"\b银\b", "人"),
    ]

    for pattern, repl in replacements:
        q = re.sub(pattern, repl, q)

    return q


conversation_state = ConversationState()


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


def redact_sensitive_text(text: str) -> str:
    t = text or ""
    t = re.sub(r"\b\d{17}[\dXx]\b", "[身份证号已脱敏]", t)
    t = re.sub(r"\b1[3-9]\d{9}\b", "[手机号已脱敏]", t)
    t = re.sub(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "[邮箱已脱敏]", t)
    t = re.sub(r"\b\d{16,19}\b", "[长数字已脱敏]", t)
    return t




def extract_strong_terms_from_question(question: str) -> list[str]:
    q = (question or "").strip()
    if not q:
        return []

    candidates = []

    # 先抓日期/阶段类短语
    candidates.extend(re.findall(r"\d{1,2}号之后", q))
    candidates.extend(re.findall(r"\d{1,2}日之后", q))
    candidates.extend(re.findall(r"\d{1,2}月\d{1,2}日", q))
    candidates.extend(re.findall(r"\d{4}年\d{1,2}月\d{1,2}日", q))

    # 再抓普通中文块
    candidates.extend(re.findall(r"[\u4e00-\u9fa5]{2,}", q))

    weak_terms = {
        "给我", "帮我", "我想", "我先", "请你",
        "分析", "整理", "梳理", "总结", "看看", "看下", "说说", "讲讲",
        "事情", "情况", "问题", "内容", "动作", "方面", "东西",
        "一下", "一下吧", "吧", "吗", "呢", "呀", "啊",
        "更详细的", "详细的", "详细点", "更详细", "详细一些",
    }

    result = []
    for c in candidates:
        c = c.strip()
        if not c:
            continue
        if c in weak_terms:
            continue
        if len(c) > 12:
            continue
        if c not in result:
            result.append(c)

    return result


def merge_rewritten_query_with_strong_terms(question: str, rewritten_query: str) -> str:
    rewritten_terms = [x.strip() for x in (rewritten_query or "").split() if x.strip()]
    strong_terms = extract_strong_terms_from_question(question)

    # 先保留 rewrite 结果，再补 question 里的强词
    merged = []

    for term in rewritten_terms + strong_terms:
        if not term:
            continue
        if term not in merged:
            merged.append(term)

    result = " ".join(merged).strip()

    # 兜底：绝不把已有 rewrite 结果清空
    if not result:
        return (rewritten_query or "").strip()

    return result


def strip_structured_request_words(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # 只剥“句式壳”，不碰核心实体
    t = re.sub(r"^(给我|帮我|请你|麻烦你|我想|我先|先)\s*", "", t)
    t = re.sub(r"(吧|吗|呢|呀|啊)$", "", t)

    # 单独一句“更详细的/详细点”这类，直接清空，避免污染
    if t in {"更详细的", "详细的", "详细点", "更详细", "详细一些"}:
        return ""

    return re.sub(r"\s+", " ", t).strip()


def build_clean_merged_query(event_merged_query: str, current_question: str) -> str:
    parent = strip_structured_request_words(event_merged_query)
    current = strip_structured_request_words(current_question)

    if parent and current:
        return f"{parent} {current}".strip()
    if current:
        return current
    if parent:
        return parent
    return (current_question or "").strip()


def is_abstract_query(question: str) -> bool:
    terms = extract_strong_terms_from_question(question)
    if not terms:
        return True

    weak_terms = {
        "事情", "情况", "问题", "内容", "性质", "方面",
        "更详细的", "详细的", "详细点", "更详细", "详细一些",
    }

    return all(t in weak_terms for t in terms)
def strip_structured_request_words(text: str) -> str:
    t = (text or "").strip()

    t = re.sub(r"(时间线|时间顺序|整理一下|梳理一下|分析一下|总结一下)", " ", t)

    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_clean_merged_query(event_merged_query: str, current_question: str) -> str:
    parent = strip_structured_request_words(event_merged_query)
    current = strip_structured_request_words(current_question)

    # 如果 parent 被剥得太空，就退回当前问题
    if not parent:
        return current or current_question

    # 如果 current 被剥空，也至少保留原问题
    if not current:
        current = current_question

    merged = f"{parent} {current}".strip()
    merged = re.sub(r"\s+", " ", merged)
    return merged


def extract_timeline_evidence_from_chunks(relevant_indices, repo_state):
    import re

    date_patterns = [
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d{1,2}月\d{1,2}日",
        r"\d{1,2}日",
        r"\d{1,2}:\d{2}",
    ]

    results = []

    for idx in relevant_indices:
        text = repo_state.chunk_texts[idx]
        path = repo_state.chunk_paths[idx]

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            if any(re.search(p, line) for p in date_patterns):
                results.append((path, line))

    # 去重
    seen = set()
    deduped = []
    for path, line in results:
        key = (path, line)
        if key not in seen:
            seen.add(key)
            deduped.append((path, line))

    return deduped
def build_timeline_evidence_text(timeline_items):
    if not timeline_items:
        return ""

    lines = []

    for path, line in timeline_items[:80]:
        lines.append(f"{path} | {line}")

    return "\n".join(lines) + "\n\n"


def inject_followup_anchor(query: str, conversation_state) -> str:
    prev_q = conversation_state.last_content_user_question or ""
    if not prev_q:
        return query

    prev_terms = extract_strong_terms_from_question(prev_q)
    curr_terms = extract_strong_terms_from_question(query)


    weak_current = {"公司", "事情", "情况", "问题", "内容"}
    filtered_curr = [t for t in curr_terms if t not in weak_current]


    if len(filtered_curr) <= 1:
        merged = prev_terms + curr_terms
    else:
        merged = prev_terms[:2] + curr_terms  # 只取前2个锚点，避免污染

    return " ".join(dict.fromkeys(merged))


def is_abstract_query(question: str) -> bool:
    terms = extract_strong_terms_from_question(question)

    if not terms:
        return True

    # ❗必须全是弱词才算抽象
    weak_terms = {"事情", "情况", "问题", "内容", "性质"}

    return all(t in weak_terms for t in terms)


def run_chat_loop(repo_state, model_emb, client, model_id: str, ollama_api_url: str, ollama_model: str, logger):
    global conversation_state

    chat_config = types.GenerateContentConfig(
        system_instruction=(
            "你是用户的个人笔记助理，擅长根据给定材料回答问题、整理线索、做有限归纳。\n"
            f"当前知识库共有 {len(repo_state.paths)} 个文件。\n"
            f"时间跨度参考：最早文件是 {repo_state.earliest_note}；最新文件是 {repo_state.latest_note}。\n"
            "你只能依据本轮给出的参考片段回答，不要假设自己看过所有文件全文。\n"
            "有证据就答，没有证据就明确说信息不足。\n"
            "不要因为名字相似、简称相似，就擅自把不同人物、公司、项目混为一谈。\n"
        ),
        temperature=0.4,
    )

    memory_buffer = []
    current_focus_file = None

    print("=================================")
    print("🤖：你好！我是你的 DocMind 随身助理。你可以问我任何问题。")

    while True:
        question = input("\n问：")
        if question.strip().lower() in ["q", "quit", "exit"]:
            break
        if not question.strip():
            continue

        question = normalize_colloquial_question(question)
        start_qa = time.time()

        try:
            from ai.query_router import route_question

            prev_content_user_question = conversation_state.last_content_user_question

            event = detect_dialog_event(question, conversation_state)
            conversation_state = apply_event_to_state(conversation_state, event)

            if event.route_hint:
                route = event.route_hint
                logger.info(f"🧭 [状态机事件] {event.name}: {question} -> {route}")
            else:
                route_info = route_question(question, ollama_api_url, ollama_model, logger)
                route = route_info["route"]
                logger.info(f"🧭 [模型路由] 问题: {question} -> {route}")

            if route == "system_capability":
                local_answer = answer_system_capability_question(question)
                if local_answer:
                    print(f"\nAI回答：\n{local_answer}")
                    clean_reply = local_answer.replace("\n", " ").replace("*", "").replace("#", "")
                    memory_buffer.extend([
                        f"用户：{question}",
                        f"AI：{clean_reply[:150] + ('...' if len(clean_reply) > 150 else '')}",
                    ])

                    conversation_state.mode = "smalltalk"
                    conversation_state.last_user_question = question
                    conversation_state.last_route = "system_capability"
                    conversation_state.last_local_topic = None
                    conversation_state.last_answer_preview = clean_reply[:200]

                    print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")
                    continue

            if route == "repo_meta":
                logger.info("📦 命中 repo_meta，准备本地回答")
                local_answer, local_topic = answer_repo_meta_question(
                    question,
                    repo_state,
                    last_user_question=prev_content_user_question,
                    topic_summarizer=lambda prompt: _call_local_ollama(
                        prompt,
                        logger=logger,
                        ollama_api_url=ollama_api_url,
                        ollama_model=ollama_model,
                    ),
                )
                logger.info(f"📦 repo_meta 返回值: {repr(local_answer)[:200]} | topic={local_topic}")

                if local_answer:
                    print(f"\nAI回答：\n{local_answer}")
                    clean_reply = local_answer.replace("\n", " ").replace("*", "").replace("#", "")
                    memory_buffer.extend([
                        f"用户：{question}",
                        f"AI：{clean_reply[:150] + ('...' if len(clean_reply) > 150 else '')}",
                    ])

                    conversation_state.mode = "repo_meta"
                    conversation_state.last_user_question = question
                    conversation_state.last_route = "repo_meta"
                    conversation_state.last_local_topic = local_topic
                    conversation_state.last_answer_preview = clean_reply[:200]
                    conversation_state.last_content_user_question = question
                    conversation_state.last_content_route = "repo_meta"
                    conversation_state.last_content_topic = local_topic

                    print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")
                    continue

            if route == "smalltalk":
                local_answer = answer_smalltalk(question, dialog_state=conversation_state)
                if local_answer:
                    print(f"\nAI回答：\n{local_answer}")
                    clean_reply = local_answer.replace("\n", " ").replace("*", "").replace("#", "")
                    memory_buffer.extend([
                        f"用户：{question}",
                        f"AI：{clean_reply[:150] + ('...' if len(clean_reply) > 150 else '')}",
                    ])

                    conversation_state.mode = "smalltalk"
                    conversation_state.last_user_question = question
                    conversation_state.last_route = "smalltalk"
                    conversation_state.last_local_topic = None
                    conversation_state.last_answer_preview = clean_reply[:200]

                    print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")
                    continue

            # 到这里默认进入 normal_retrieval
            conversation_state.mode = "content"
            conversation_state.last_user_question = question
            conversation_state.last_route = "normal_retrieval"
            conversation_state.last_local_topic = None

            flags = determine_query_flags(question)

            if flags["skip_retrieval"] or flags["is_inventory_query"]:
                search_query = question

            else:
                base_query = question

                if event.merged_query:
                    clean_merged_query = build_clean_merged_query(event.merged_query, question)
                    logger.info(f"🧩 [状态机拼接检索] raw={event.merged_query}")
                    logger.info(f"🧼 [模式词去壳后] merged={clean_merged_query}")
                    base_query = clean_merged_query

                # 抽象追问：优先继承上一轮有效检索词，而不是继承“更详细的”这种原话
                if is_abstract_query(question):
                    prev_anchor = (conversation_state.last_effective_search_query or "").strip()
                    if prev_anchor:
                        base_query = f"{prev_anchor} {question}".strip()
                        logger.info(f"🪝 [抽象追问继承] anchor={prev_anchor}")

                raw_search_query = rewrite_search_query(
                    base_query,
                    memory_buffer,
                    ollama_api_url,
                    ollama_model,
                    logger,
                )

                search_query = merge_rewritten_query_with_strong_terms(question, raw_search_query)

                # 再兜底一次：如果 merge 后反而空了，就退回 rewrite 结果；rewrite 也空就退回 base_query
                if not search_query.strip():
                    search_query = (raw_search_query or base_query or question).strip()

                logger.info(f"🛡️ [强词保底后]：{search_query}")
            context_text = ""
            inventory_candidates_text = (
                build_inventory_candidates_text(question, repo_state, flags["inventory_target_type"])
                if flags["is_inventory_query"]
                else ""
            )
            timeline_evidence_text = ""
            if not flags["skip_retrieval"]:
                if search_query.strip():
                    conversation_state.last_effective_search_query = search_query.strip()
                retrieval = perform_retrieval(
                    question,
                    search_query,
                    repo_state,
                    model_emb,
                    logger,
                    current_focus_file,
                )
                current_focus_file = retrieval["current_focus_file"]
                context_text = build_context_text(retrieval["relevant_indices"], repo_state, logger)
                timeline_evidence_text = ""

                if any(x in question for x in ["时间线", "经过", "过程", "梳理", "更详细", "详细点"]):
                    timeline_items = extract_timeline_evidence_from_chunks(
                        retrieval["relevant_indices"],
                        repo_state,
                    )
                    timeline_evidence_text = build_timeline_evidence_text(timeline_items)
            if route == "normal_retrieval" and not context_text.strip() and not inventory_candidates_text.strip():
                fallback_answer = "这次没有检索到足够可靠的参考片段，所以我先不乱回答。你可以再具体一点。"
                print(f"\nAI回答：\n{fallback_answer}")

                clean_reply = fallback_answer.replace("\n", " ").replace("*", "").replace("#", "")
                memory_buffer.extend([
                    f"用户：{question}",
                    f"AI：{clean_reply[:150] + ('...' if len(clean_reply) > 150 else '')}",
                ])

                conversation_state.last_answer_preview = clean_reply[:200]
                conversation_state.last_content_user_question = question
                conversation_state.last_content_route = "normal_retrieval"
                conversation_state.last_content_topic = None

                print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")
                continue

            safe_memory_buffer = [redact_sensitive_text(x) for x in memory_buffer]
            safe_inventory_candidates_text = redact_sensitive_text(inventory_candidates_text)
            safe_context_text = redact_sensitive_text(timeline_evidence_text + context_text)
            safe_question = redact_sensitive_text(question)

            final_prompt = build_final_prompt(
                memory_buffer=safe_memory_buffer,
                current_focus_file=current_focus_file,
                inventory_candidates_text=safe_inventory_candidates_text,
                context_text=safe_context_text,
                question=safe_question,
            )

            response = client.models.generate_content(
                model=model_id,
                contents=final_prompt,
                config=chat_config,
            )

            if response.text:
                print(f"\nAI回答：\n{response.text}")
                clean_reply = response.text.replace("\n", " ").replace("*", "").replace("#", "")
                memory_buffer.extend([
                    f"用户：{question}",
                    f"AI：{clean_reply[:150] + ('...' if len(clean_reply) > 150 else '')}",
                ])
                conversation_state.last_answer_preview = clean_reply[:200]

            # normal_retrieval 成功后，再更新内容态
            conversation_state.last_content_user_question = question
            conversation_state.last_content_route = "normal_retrieval"
            conversation_state.last_content_topic = None

            print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")

        except Exception as e:
            logger.error(f"\n调用失败: {e}")