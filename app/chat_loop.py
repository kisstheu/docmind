from __future__ import annotations

from pathlib import Path

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
from app.dialog_state_machine import extract_result_set_from_answer
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
from app.chat_text_utils import maybe_build_related_records_answer
from app.file_actions.loop import handle_file_action_turn
from infra.file_change_store import FileChangeStore


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

    if local_topic == "time" and "份 Word 文档是" in local_answer:  # 针对时间查询的列表
        items, entity_type = extract_result_set_from_answer(local_answer, "文件")
        conversation_state.last_result_set_items = items
        conversation_state.last_result_set_entity_type = entity_type
        conversation_state.last_result_set_query = question  # 可选，记录原查询
        logger.info(f"🗂️ [结果集提取] 捕获 {len(items)} 个项: {items}")

    return local_answer, local_topic


def try_handle_smalltalk(route: str, question: str) -> str | None:
    global conversation_state

    if route != "smalltalk":
        return None
    return answer_smalltalk(question, dialog_state=conversation_state)


def run_chat_loop(
    repo_state,
    model_emb,
    client,
    model_id: str,
    ollama_api_url: str,
    ollama_model: str,
    logger,
    *,
    notes_dir: Path,
    change_log_file: Path,
):
    global conversation_state
    change_store = FileChangeStore(change_log_file)

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

            file_action_handled, conversation_state, current_focus_file = handle_file_action_turn(
                question=question,
                start_qa=start_qa,
                state=conversation_state,
                memory_buffer=memory_buffer,
                current_focus_file=current_focus_file,
                repo_state=repo_state,
                notes_dir=notes_dir,
                change_store=change_store,
            )
            if file_action_handled:
                continue

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

            related_records_answer = maybe_build_related_records_answer(
                question=question,
                relevant_indices=last_relevant_indices,
                repo_state=repo_state,
            )
            if related_records_answer:
                logger.info("🧩 [相关记录稳态回答] 使用本地列举，避免生成阶段漏列")
                print_answer(related_records_answer, start_qa)
                append_memory(memory_buffer, question, related_records_answer)
                conversation_state = update_state_after_retrieval_answer(
                    conversation_state,
                    question,
                    related_records_answer,
                    logger,
                )
                continue

            if (
                route == "normal_retrieval"
                and not materials["context_text"].strip()
                and not materials["inventory_candidates_text"].strip()
            ):
                if not list(getattr(repo_state, "paths", []) or []):
                    fallback_answer = "当前知识库还没有可检索文档。请先在笔记目录中放入支持的文件（如 .md/.txt/.pdf/.png 等），再来提问。"
                elif not list(getattr(repo_state, "chunk_texts", []) or []):
                    fallback_answer = "当前文档还没有形成可检索片段，请先检查文件读取/索引是否成功。"
                else:
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
