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
from app.dialog_utils import (
    is_followup_question,
    is_judgment_request,
    is_repo_meta_confirmation,
    is_action_request,
    is_content_followup_question,
    is_relationship_analysis_request,
    is_query_correction, is_structured_output_request,
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

            forced_route = None

            prev_content_user_question = dialog_state.get("last_content_user_question")
            prev_content_route = dialog_state.get("last_content_route")
            last_local_topic = dialog_state.get("last_local_topic")

            # 先判断是否属于“查询纠偏”
            is_correction = bool(is_query_correction(question) and prev_content_user_question)

            if is_correction:
                forced_route = "normal_retrieval"
                logger.info(f"🧭 [查询纠偏] 问题: {question} -> normal_retrieval")
            elif is_structured_output_request(question):
                forced_route = "normal_retrieval"
                logger.info(f"🧭 [结构化请求] 问题: {question} -> normal_retrieval")
            elif is_relationship_analysis_request(question):
                forced_route = "normal_retrieval"
                logger.info(f"🧭 [关系判断] 问题: {question} -> normal_retrieval")

            elif is_action_request(question):
                forced_route = "normal_retrieval"
                logger.info(f"🧭 [动作意图] 问题: {question} -> normal_retrieval")

            elif is_judgment_request(question):
                forced_route = "normal_retrieval"
                logger.info(f"🧭 [语义判断] 问题: {question} -> normal_retrieval")

            elif prev_content_route == "repo_meta":
                if is_followup_question(question):
                    forced_route = "repo_meta"
                    logger.info(f"🧭 [追问继承] 问题: {question} -> repo_meta")

                elif (
                    last_local_topic in {"category", "category_summary", "category_confirm"}
                    and is_repo_meta_confirmation(question)
                ):
                    forced_route = "repo_meta"
                    logger.info(f"🧭 [分类确认继承] 问题: {question} -> repo_meta")

                elif (
                    len(question.strip()) <= 12
                    and any(x in question for x in ["粗", "细", "大类", "概括", "方面", "分类"])
                ):
                    forced_route = "repo_meta"
                    logger.info(f"🧭 [短句语义继承] 问题: {question} -> repo_meta")

            elif prev_content_route == "normal_retrieval":
                if is_content_followup_question(question):
                    forced_route = "normal_retrieval"
                    logger.info(f"🧭 [内容追问继承] 问题: {question} -> normal_retrieval")

            if forced_route:
                route = forced_route
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

                    print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")
                    continue

            # 到这里默认进入 normal_retrieval
            dialog_state["last_user_question"] = question
            dialog_state["last_route"] = "normal_retrieval"
            dialog_state["last_local_topic"] = None

            flags = determine_query_flags(question)

            is_retrieval_followup = (
                prev_content_route == "normal_retrieval"
                and is_content_followup_question(question)
            )

            if flags["skip_retrieval"] or flags["is_inventory_query"]:
                search_query = question

            elif is_correction and prev_content_user_question:
                merged_q = f"{prev_content_user_question} {question}"
                logger.info(f"🧩 [纠偏拼接检索] parent={prev_content_user_question} | current={question}")
                search_query = rewrite_search_query(
                    merged_q,
                    memory_buffer,
                    ollama_api_url,
                    ollama_model,
                    logger,
                )

            elif is_retrieval_followup and prev_content_user_question:
                merged_q = f"{prev_content_user_question} {question}"
                logger.info(f"🧩 [追问拼接检索] parent={prev_content_user_question} | current={question}")
                search_query = rewrite_search_query(
                    merged_q,
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

            print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")

        except Exception as e:
            logger.error(f"\n调用失败: {e}")