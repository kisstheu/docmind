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


dialog_state = {
    "last_user_question": None,
    "last_route": None,
    "last_local_topic": None,
    "last_answer_preview": None,
    "last_content_user_question": None,
    "last_content_route": None,
    "last_content_topic": None,
}

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

            prev_content_user_question = dialog_state.get("last_content_user_question")
            prev_content_route = dialog_state.get("last_content_route")

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

                    dialog_state["last_user_question"] = question
                    dialog_state["last_route"] = "system_capability"
                    dialog_state["last_local_topic"] = None
                    dialog_state["last_answer_preview"] = clean_reply[:200]

                    conversation_state.last_route = "system_capability"
                    conversation_state.last_local_topic = None

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

                    dialog_state["last_user_question"] = question
                    dialog_state["last_route"] = "repo_meta"
                    dialog_state["last_local_topic"] = local_topic
                    dialog_state["last_answer_preview"] = clean_reply[:200]
                    dialog_state["last_content_user_question"] = question
                    dialog_state["last_content_route"] = "repo_meta"
                    dialog_state["last_content_topic"] = local_topic

                    conversation_state.mode = "repo_meta"
                    conversation_state.last_route = "repo_meta"
                    conversation_state.last_local_topic = local_topic
                    conversation_state.last_content_user_question = question
                    conversation_state.last_content_route = "repo_meta"
                    conversation_state.last_content_topic = local_topic

                    print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")
                    continue

            if route == "smalltalk":
                local_answer = answer_smalltalk(question, dialog_state=dialog_state)
                if local_answer:
                    print(f"\nAI回答：\n{local_answer}")
                    clean_reply = local_answer.replace("\n", " ").replace("*", "").replace("#", "")
                    memory_buffer.extend([
                        f"用户：{question}",
                        f"AI：{clean_reply[:150] + ('...' if len(clean_reply) > 150 else '')}",
                    ])

                    dialog_state["last_user_question"] = question
                    dialog_state["last_route"] = "smalltalk"
                    dialog_state["last_local_topic"] = None
                    dialog_state["last_answer_preview"] = clean_reply[:200]

                    conversation_state.mode = "smalltalk"
                    conversation_state.last_route = "smalltalk"
                    conversation_state.last_local_topic = None

                    print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")
                    continue

            # 到这里默认进入 normal_retrieval
            dialog_state["last_user_question"] = question
            dialog_state["last_route"] = "normal_retrieval"
            dialog_state["last_local_topic"] = None

            conversation_state.mode = "content"
            conversation_state.last_route = "normal_retrieval"
            conversation_state.last_local_topic = None

            flags = determine_query_flags(question)

            if flags["skip_retrieval"] or flags["is_inventory_query"]:
                search_query = question
            elif event.merged_query:
                logger.info(f"🧩 [状态机拼接检索] merged={event.merged_query}")
                search_query = rewrite_search_query(
                    event.merged_query,
                    memory_buffer,
                    ollama_api_url,
                    ollama_model,
                    logger,
                )
            else:
                search_query = rewrite_search_query(
                    question,
                    memory_buffer,
                    ollama_api_url,
                    ollama_model,
                    logger,
                )

            context_text = ""
            inventory_candidates_text = (
                build_inventory_candidates_text(question, repo_state, flags["inventory_target_type"])
                if flags["is_inventory_query"]
                else ""
            )

            if not flags["skip_retrieval"]:
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

            if route == "normal_retrieval" and not context_text.strip() and not inventory_candidates_text.strip():
                fallback_answer = "这次没有检索到足够可靠的参考片段，所以我先不乱回答。你可以再具体一点。"
                print(f"\nAI回答：\n{fallback_answer}")

                clean_reply = fallback_answer.replace("\n", " ").replace("*", "").replace("#", "")
                memory_buffer.extend([
                    f"用户：{question}",
                    f"AI：{clean_reply[:150] + ('...' if len(clean_reply) > 150 else '')}",
                ])

                dialog_state["last_answer_preview"] = clean_reply[:200]
                dialog_state["last_content_user_question"] = question
                dialog_state["last_content_route"] = "normal_retrieval"
                dialog_state["last_content_topic"] = None

                conversation_state.last_content_user_question = question
                conversation_state.last_content_route = "normal_retrieval"
                conversation_state.last_content_topic = None

                print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")
                continue

            safe_memory_buffer = [redact_sensitive_text(x) for x in memory_buffer]
            safe_inventory_candidates_text = redact_sensitive_text(inventory_candidates_text)
            safe_context_text = redact_sensitive_text(context_text)
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
                dialog_state["last_answer_preview"] = clean_reply[:200]

            # normal_retrieval 成功后，再更新内容态
            dialog_state["last_content_user_question"] = question
            dialog_state["last_content_route"] = "normal_retrieval"
            dialog_state["last_content_topic"] = None

            conversation_state.last_content_user_question = question
            conversation_state.last_content_route = "normal_retrieval"
            conversation_state.last_content_topic = None

            print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")

        except Exception as e:
            logger.error(f"\n调用失败: {e}")