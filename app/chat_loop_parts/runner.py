from __future__ import annotations

import time
from pathlib import Path

from ai.repo_meta.category import resolve_repo_content_category_scope
from ai.structured_skill_summary import summarize_structured_skill_summary_with_remote
from ai.decision_result import parse_decision_result, render_decision_result
from app.dialog_state_machine import apply_event_to_state, detect_dialog_event
from app.chat_retrieval_flow import (
    build_retrieval_materials,
    build_safe_final_prompt,
    build_search_query,
    resolve_route,
)
from app.chat_state_helpers import (
    append_memory,
    print_answer,
    update_state_after_local_answer,
    update_state_after_retrieval_answer,
)
from app.chat_text.core import normalize_colloquial_question
from app.chat_text.file_lookup import maybe_build_file_location_answer
from app.chat_text.file_lookup import looks_like_focused_file_content_question
from app.chat_text.lookup_answer_main import maybe_build_direct_lookup_answer
from app.chat_text.related_records import maybe_build_related_records_answer
from app.file_actions.loop import handle_file_action_turn
from infra.file_change_store import FileChangeStore
from retrieval.search_engine import determine_query_flags
import app.chat_loop_handlers as _loop_handlers


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
    question_recorder=None,
):
    from app import chat_loop as runtime
    change_store = FileChangeStore(change_log_file)
    chat_config = runtime.build_chat_config(repo_state)
    memory_buffer: list[str] = []
    current_focus_file = None
    last_relevant_indices = []
    is_first_turn = True
    print("=================================")
    print("🤖：你好！我是你的 DocMind 随身助理。你可以问我任何问题。")
    while True:
        if is_first_turn:
            runtime._flush_pending_tty_input_unix()
        raw_question = runtime._read_user_question(use_fresh_tty_input=is_first_turn)
        if is_first_turn:
            is_first_turn = False
        if raw_question.strip().lower() in ["q", "quit", "exit"]:
            break
        if not raw_question.strip():
            continue
        question = normalize_colloquial_question(raw_question)
        if question_recorder is not None:
            question_recorder.record(raw_question, normalized_question=question)
        try:
            start_qa = time.time()
            file_action_handled, runtime.conversation_state, current_focus_file = handle_file_action_turn(
                question=question,
                start_qa=start_qa,
                state=runtime.conversation_state,
                memory_buffer=memory_buffer,
                current_focus_file=current_focus_file,
                repo_state=repo_state,
                model_emb=model_emb,
                notes_dir=notes_dir,
                change_store=change_store,
            )
            if file_action_handled:
                continue
            prev_content_user_question = runtime.conversation_state.last_content_user_question
            event = detect_dialog_event(
                question,
                runtime.conversation_state,
                logger,
                focused_file=current_focus_file,
            )
            local_answer = runtime.try_handle_contextless_followup(
                question=question,
                state=runtime.conversation_state,
                event=event,
                logger=logger,
            )
            if local_answer is not None:
                print_answer(local_answer, start_qa)
                append_memory(memory_buffer, question, local_answer)
                runtime.conversation_state = update_state_after_local_answer(
                    runtime.conversation_state,
                    question=question,
                    answer=local_answer,
                    route="normal_retrieval",
                    local_topic=None,
                    is_content_answer=False,
                )
                continue
            runtime.conversation_state = apply_event_to_state(runtime.conversation_state, event)
            route_info = resolve_route(question, event, ollama_api_url, ollama_model, logger, state=runtime.conversation_state)
            route = route_info["route"]
            prefetched_smalltalk_answer = route_info.get("smalltalk_reply", "")
            route_question_input = route_info.get("route_question_input", question)
            # 1) system capability
            local_answer = runtime.try_handle_system_capability(route, question)
            if local_answer is not None:
                print_answer(local_answer, start_qa)
                append_memory(memory_buffer, question, local_answer)
                runtime.conversation_state = update_state_after_local_answer(
                    runtime.conversation_state,
                    question=question,
                    answer=local_answer,
                    route="system_capability",
                    local_topic=None,
                    is_content_answer=False,
                )
                continue
            # 2) repo meta
            local_answer, local_topic = runtime.try_handle_repo_meta(
                route,
                question,
                repo_state,
                model_emb,
                logger,
                prev_content_user_question,
                ollama_api_url,
                ollama_model,
                conversation_state=runtime.conversation_state,
                client=client,
                model_id=model_id,
            )
            if local_answer is not None:
                print_answer(local_answer, start_qa)
                append_memory(memory_buffer, question, local_answer)
                runtime.conversation_state = update_state_after_local_answer(
                    runtime.conversation_state,
                    question=question,
                    answer=local_answer,
                    route="repo_meta",
                    local_topic=local_topic,
                    is_content_answer=True,
                )
                continue
            # 3) smalltalk
            local_answer = runtime.try_handle_smalltalk(
                route=route,
                question=question,
                ollama_api_url=ollama_api_url,
                ollama_model=ollama_model,
                logger=logger,
                conversation_state=runtime.conversation_state,
                prefetched_smalltalk_answer=prefetched_smalltalk_answer,
            )
            if local_answer is not None:
                print_answer(local_answer, start_qa)
                append_memory(memory_buffer, question, local_answer)
                runtime.conversation_state = update_state_after_local_answer(
                    runtime.conversation_state,
                    question=question,
                    answer=local_answer,
                    route="smalltalk",
                    local_topic=None,
                    is_content_answer=False,
                )
                continue
            # 4) out_of_scope
            local_answer = runtime.try_handle_out_of_scope(
                route=route,
                question=question,
                ollama_api_url=ollama_api_url,
                ollama_model=ollama_model,
                logger=logger,
                conversation_state=runtime.conversation_state,
                effective_question=route_question_input,
            )
            if local_answer is not None:
                print_answer(local_answer, start_qa)
                append_memory(memory_buffer, question, local_answer)
                runtime.conversation_state = update_state_after_local_answer(
                    runtime.conversation_state,
                    question=question,
                    answer=local_answer,
                    route="out_of_scope",
                    local_topic=None,
                    is_content_answer=False,
                )
                continue
            # 5) normal retrieval
            runtime.conversation_state.mode = "content"
            runtime.conversation_state.last_user_question = question
            runtime.conversation_state.last_route = "normal_retrieval"
            runtime.conversation_state.last_local_topic = None
            result_set_summary_answer = _loop_handlers._try_answer_file_result_set_topic_summary(
                question=question,
                event_name=event.name,
                repo_state=repo_state,
                conversation_state=runtime.conversation_state,
                model_emb=model_emb,
                logger=logger,
                ollama_api_url=ollama_api_url,
                ollama_model=ollama_model,
            )
            if result_set_summary_answer:
                print_answer(result_set_summary_answer, start_qa)
                append_memory(memory_buffer, question, result_set_summary_answer)
                runtime.conversation_state = update_state_after_retrieval_answer(
                    runtime.conversation_state,
                    question,
                    result_set_summary_answer,
                    logger,
                    event_name=event.name,
                )
                continue
            flags = determine_query_flags(question)
            analytic_retrieval = _loop_handlers.looks_like_analytic_retrieval_question(
                question,
                has_collection_context=bool(runtime.conversation_state.last_result_set_items),
                has_selected_candidate=bool(runtime.conversation_state.last_selected_candidate),
            )
            category_scope_label = None
            category_scope_paths = None
            if runtime.conversation_state.last_category_context_answer:
                category_scope_label, resolved_scope_paths = resolve_repo_content_category_scope(
                    question=question,
                    repo_state=repo_state,
                    previous_summary=runtime.conversation_state.last_category_context_answer,
                    last_local_answer=(runtime.conversation_state.last_answer_text or runtime.conversation_state.last_answer_preview),
                    model_emb=model_emb,
                )
                if resolved_scope_paths:
                    category_scope_paths = resolved_scope_paths
                    logger.info(
                        f"[板块范围继承] {category_scope_label} -> {len(category_scope_paths)} files"
                    )
            search_query, context_anchor = build_search_query(
                question=question,
                event=event,
                flags=flags,
                memory_buffer=memory_buffer,
                last_effective_search_query=runtime.conversation_state.last_effective_search_query,
                last_user_question=runtime.conversation_state.last_content_user_question,
                last_answer_type=runtime.conversation_state.last_answer_type,
                last_result_set_items=runtime.conversation_state.last_result_set_items,
                last_result_set_entity_type=runtime.conversation_state.last_result_set_entity_type,
                last_selected_candidate=runtime.conversation_state.last_selected_candidate,
                last_selected_source_files=runtime.conversation_state.last_selected_source_files,
                last_relevant_indices=last_relevant_indices,
                logger=logger,
                ollama_api_url=ollama_api_url,
                ollama_model=ollama_model,
            )
            if not flags["skip_retrieval"] and search_query.strip():
                runtime.conversation_state.last_effective_search_query = search_query.strip()
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
                allowed_paths=category_scope_paths,
                scope_label=category_scope_label,
                selected_source_files=runtime.conversation_state.last_selected_source_files,
            )
            current_focus_file = materials["current_focus_file"]
            last_relevant_indices = materials["relevant_indices"]
            focused_file_content_followup = bool(
                current_focus_file and looks_like_focused_file_content_question(question)
            )
            related_records_answer = maybe_build_related_records_answer(
                question=question,
                relevant_indices=last_relevant_indices,
                repo_state=repo_state,
            )
            if related_records_answer:
                logger.info("🧩 [相关记录稳态回答] 使用本地列举，避免生成阶段漏列")
                print_answer(related_records_answer, start_qa)
                append_memory(memory_buffer, question, related_records_answer)
                runtime.conversation_state = update_state_after_retrieval_answer(
                    runtime.conversation_state,
                    question,
                    related_records_answer,
                    logger,
                    event_name=event.name,
                )
                continue
            local_file_locator_answer = None if (analytic_retrieval or focused_file_content_followup) else maybe_build_file_location_answer(
                question=question,
                search_query=search_query,
                relevant_indices=last_relevant_indices,
                repo_state=repo_state,
                logger=logger,
                allow_followup_inference=(
                    runtime.conversation_state.last_answer_type == "enumeration_file"
                    and event.name in {"content_followup", "result_set_followup", "result_set_expansion_followup"}
                ),
            )
            if local_file_locator_answer:
                logger.info("🧩 [文件定位稳态回答] 使用本地规则直接回答，跳过远程模型生成")
                print_answer(local_file_locator_answer, start_qa)
                append_memory(memory_buffer, question, local_file_locator_answer)
                runtime.conversation_state = update_state_after_retrieval_answer(
                    runtime.conversation_state,
                    question,
                    local_file_locator_answer,
                    logger,
                    event_name=event.name,
                )
                continue
            local_entity_mapping_answer = None if (analytic_retrieval or focused_file_content_followup) else maybe_build_direct_lookup_answer(
                question=question,
                search_query=search_query,
                relevant_indices=last_relevant_indices,
                repo_state=repo_state,
                logger=logger,
                allow_followup_inference=(
                    (
                        (
                            bool(runtime.conversation_state.last_result_set_items)
                            and str(runtime.conversation_state.last_answer_type or "").startswith("enumeration_")
                        )
                        or (
                            "可直接核对的证据"
                            in str(runtime.conversation_state.last_answer_text or runtime.conversation_state.last_answer_preview or "")
                        )
                    )
                    and event.name in {
                        "unknown",
                        "content_followup",
                        "result_set_followup",
                        "result_set_expansion_followup",
                        "entity_lookup_followup",
                    }
                ),
            )
            if local_entity_mapping_answer:
                logger.info("🧩 [直接检索稳态回答] 使用本地证据抽取，跳过远程模型生成")
                print_answer(local_entity_mapping_answer, start_qa)
                append_memory(memory_buffer, question, local_entity_mapping_answer)
                runtime.conversation_state = update_state_after_retrieval_answer(
                    runtime.conversation_state,
                    question,
                    local_entity_mapping_answer,
                    logger,
                    event_name=event.name,
                )
                continue
            fallback_local_answer = runtime.try_handle_retrieval_force_local_or_empty_context(
                route=route,
                question=question,
                event_name=event.name,
                search_query=search_query,
                relevant_indices=last_relevant_indices,
                repo_state=repo_state,
                materials=materials,
                conversation_state=runtime.conversation_state,
                model_emb=model_emb,
                logger=logger,
                ollama_api_url=ollama_api_url,
                ollama_model=ollama_model,
                prefer_content_answer=focused_file_content_followup,
            )
            if fallback_local_answer:
                if event.name == "structured_skill_summary":
                    fallback_local_answer = summarize_structured_skill_summary_with_remote(
                        materials_markdown=fallback_local_answer,
                        question=question,
                        client=client,
                        model_id=model_id,
                        logger=logger,
                    )
                print_answer(fallback_local_answer, start_qa)
                append_memory(memory_buffer, question, fallback_local_answer)
                runtime.conversation_state = update_state_after_retrieval_answer(
                    runtime.conversation_state,
                    question,
                    fallback_local_answer,
                    logger,
                    event_name=event.name,
                    focused_file=current_focus_file,
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
                result_set_items=runtime.conversation_state.last_result_set_items,
                selected_candidate=runtime.conversation_state.last_selected_candidate,
                selected_source_files=runtime.conversation_state.last_selected_source_files,
            )
            logger.info("🛰️ [远程模型生成] 进入生成阶段，开始调用远程大模型")
            response = client.models.generate_content(
                model=model_id,
                contents=final_prompt,
                config=chat_config,
            )
            answer_text = response.text or "这次我没有生成有效回答。"
            decision_result = None
            if event.name == "decision_request":
                decision_result = parse_decision_result(answer_text, user_question=question)
                if decision_result is not None:
                    answer_text = render_decision_result(decision_result)
                    logger.info(
                        "🧭 [结构化决策结果] "
                        f"candidate={decision_result.selected_candidate} | "
                        f"sources={list(decision_result.source_files)}"
                    )
                else:
                    logger.warning("⚠️ [结构化决策结果] 远程回答缺少可解析字段，不写入选择状态")
            print_answer(answer_text, start_qa)
            append_memory(memory_buffer, question, answer_text)
            runtime.conversation_state = update_state_after_retrieval_answer(
                runtime.conversation_state,
                question,
                answer_text,
                logger,
                event_name=event.name,
                focused_file=current_focus_file,
                decision_result=decision_result,
            )
        except Exception as e:
            err = str(e)
            if "UNEXPECTED_EOF_WHILE_READING" in err or "EOF occurred in violation of protocol" in err:
                logger.error( "模型服务连接被中途断开，可能是代理或网络波动导致，请重试。")
            logger.error(f"\n调用失败: {e}")
