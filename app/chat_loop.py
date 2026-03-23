from __future__ import annotations

from google.genai import types

from ai.capabilities import (
    answer_repo_meta_question,
    answer_smalltalk,
    answer_system_capability_question,
)
from app.dialog_state_machine import (
    ConversationState,
    apply_event_to_state,
    detect_dialog_event,
)
from retrieval.search_engine import determine_query_flags

from app.chat_retrieval_flow import (
    build_retrieval_materials,
    build_safe_final_prompt,
    build_search_query,
    build_topic_summarizer,
    resolve_route,
)
from app.chat_state_helpers import (
    append_memory,
    print_answer,
    update_state_after_local_answer,
    update_state_after_retrieval_answer,
)
from app.chat_text_utils import normalize_colloquial_question


conversation_state = ConversationState()

def try_handle_system_capability(route: str, question: str) -> str | None:
    if route != "system_capability":
        return None
    return answer_system_capability_question(question)


def try_handle_repo_meta(
    route: str,
    question: str,
    repo_state,
    model_emb,
    logger,
    prev_content_user_question: str | None,
    ollama_api_url: str,
    ollama_model: str,
):
    global conversation_state

    if route != "repo_meta":
        return None, None

    logger.info("📦 命中 repo_meta，准备本地回答")
    local_answer, local_topic = answer_repo_meta_question(
        question,
        repo_state,
        model_emb=model_emb,
        last_user_question=prev_content_user_question,
        last_local_topic=conversation_state.last_local_topic,
        topic_summarizer=build_topic_summarizer(logger, ollama_api_url, ollama_model),
    )
    logger.info(f"📦 repo_meta 返回值: {repr(local_answer)[:200]} | topic={local_topic}")
    return local_answer, local_topic


def try_handle_smalltalk(route: str, question: str) -> str | None:
    global conversation_state

    if route != "smalltalk":
        return None
    return answer_smalltalk(question, dialog_state=conversation_state)


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

    memory_buffer: list[str] = []
    current_focus_file = None
    last_relevant_indices = []

    print("=================================")
    print("🤖：你好！我是你的 DocMind 随身助理。你可以问我任何问题。")

    while True:
        question = input("\n问：")
        if question.strip().lower() in ["q", "quit", "exit"]:
            break
        if not question.strip():
            continue

        question = normalize_colloquial_question(question)

        try:
            import time
            start_qa = time.time()

            prev_content_user_question = conversation_state.last_content_user_question

            event = detect_dialog_event(question, conversation_state,logger)
            conversation_state = apply_event_to_state(conversation_state, event)

            route = resolve_route(question, event, ollama_api_url, ollama_model, logger)

            # 1) system capability
            local_answer = try_handle_system_capability(route, question)
            if local_answer is not None:
                print_answer(local_answer, start_qa)
                append_memory(memory_buffer, question, local_answer)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=local_answer,
                    route="system_capability",
                    local_topic=None,
                    is_content_answer=False,
                )
                continue

            # 2) repo meta
            local_answer, local_topic = try_handle_repo_meta(
                route,
                question,
                repo_state,
                model_emb,
                logger,
                prev_content_user_question,
                ollama_api_url,
                ollama_model,
            )
            if local_answer is not None:
                print_answer(local_answer, start_qa)
                append_memory(memory_buffer, question, local_answer)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=local_answer,
                    route="repo_meta",
                    local_topic=local_topic,
                    is_content_answer=True,
                )
                continue

            # 3) smalltalk
            local_answer = try_handle_smalltalk(route, question)
            if local_answer is not None:
                print_answer(local_answer, start_qa)
                append_memory(memory_buffer, question, local_answer)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=local_answer,
                    route="smalltalk",
                    local_topic=None,
                    is_content_answer=False,
                )
                continue

            # 4) normal retrieval
            conversation_state.mode = "content"
            conversation_state.last_user_question = question
            conversation_state.last_route = "normal_retrieval"
            conversation_state.last_local_topic = None

            flags = determine_query_flags(question)

            search_query, context_anchor = build_search_query(
                question=question,
                event=event,
                flags=flags,
                memory_buffer=memory_buffer,
                last_effective_search_query=conversation_state.last_effective_search_query,
                last_user_question=conversation_state.last_content_user_question,
                last_answer_type=conversation_state.last_answer_type,
                last_result_set_items=conversation_state.last_result_set_items,
                last_result_set_entity_type=conversation_state.last_result_set_entity_type,
                last_relevant_indices=last_relevant_indices,
                logger=logger,
                ollama_api_url=ollama_api_url,
                ollama_model=ollama_model,
            )

            if not flags["skip_retrieval"] and search_query.strip():
                conversation_state.last_effective_search_query = search_query.strip()

            materials = build_retrieval_materials(
                question=question,
                search_query=search_query,
                context_anchor=context_anchor,
                flags=flags,
                repo_state=repo_state,
                model_emb=model_emb,
                logger=logger,
                current_focus_file=current_focus_file,
                last_relevant_indices=last_relevant_indices,
                event=event,
            )
            current_focus_file = materials["current_focus_file"]
            last_relevant_indices = materials["relevant_indices"]
            if (
                route == "normal_retrieval"
                and not materials["context_text"].strip()
                and not materials["inventory_candidates_text"].strip()
            ):
                fallback_answer = "这次没有检索到足够可靠的参考片段，所以我先不乱回答。你可以再具体一点。"
                print_answer(fallback_answer, start_qa)
                append_memory(memory_buffer, question, fallback_answer)
                conversation_state = update_state_after_retrieval_answer(
                    conversation_state,
                    question,
                    fallback_answer,logger
                )
                continue

            final_prompt = build_safe_final_prompt(
                memory_buffer=memory_buffer,
                current_focus_file=current_focus_file,
                inventory_candidates_text=materials["inventory_candidates_text"],
                context_text=materials["context_text"],
                timeline_evidence_text=materials["timeline_evidence_text"],
                question=question,
                event_name=event.name,
                result_set_items=conversation_state.last_result_set_items,
            )

            response = client.models.generate_content(
                model=model_id,
                contents=final_prompt,
                config=chat_config,
            )

            answer_text = response.text or "这次我没有生成有效回答。"
            print_answer(answer_text, start_qa)
            append_memory(memory_buffer, question, answer_text)
            conversation_state = update_state_after_retrieval_answer(
                conversation_state,
                question,
                answer_text,logger
            )

        except Exception as e:
            err = str(e)
            if "UNEXPECTED_EOF_WHILE_READING" in err or "EOF occurred in violation of protocol" in err:
                logger.error( "模型服务连接被中途断开，可能是代理或网络波动导致，请重试。")
            logger.error(f"\n调用失败: {e}")