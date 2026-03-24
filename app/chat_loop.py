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
from app.file_change_history_flow import extract_history_limit, is_change_history_query
from app.file_delete_flow import (
    build_delete_preview,
    is_delete_request,
    parse_delete_confirmation_decision,
    resolve_delete_source_file,
)
from app.file_rename_flow import (
    build_rename_preview,
    extract_new_name_candidate,
    is_rename_history_query,
    is_rename_request,
    normalize_target_filename,
    parse_confirmation_decision,
    resolve_source_file,
)
from app.repo_state_mutations import apply_repo_state_delete, apply_repo_state_rename
from infra.file_change_store import FileChangeStore, collect_file_snapshot


conversation_state = ConversationState()


def _clear_pending_action_state() -> None:
    global conversation_state
    conversation_state.pending_action_type = None
    conversation_state.pending_action_source_path = None
    conversation_state.pending_action_target_path = None
    conversation_state.pending_action_requested_text = None
    conversation_state.pending_action_preview = None


def _find_repo_path_by_reference(source_ref: str | None, repo_paths: list[str]) -> str | None:
    if not source_ref:
        return None

    ref = source_ref.strip().replace("\\", "/")
    if not ref:
        return None

    ref_norm = ref.lower().replace(" ", "")
    stripped_prefixes = ("这个", "那个", "那", "该", "这", "把")
    for pfx in stripped_prefixes:
        if ref_norm.startswith(pfx):
            ref_norm = ref_norm[len(pfx):]

    for path in repo_paths:
        if path.lower().replace(" ", "") == ref_norm:
            return path

    for path in repo_paths:
        norm_path = path.lower().replace(" ", "")
        if norm_path.endswith(ref_norm):
            return path
        if ref_norm.endswith(norm_path):
            return path

    ref_name = Path(ref).name.lower().replace(" ", "")
    for path in repo_paths:
        if Path(path).name.lower().replace(" ", "") == ref_name:
            return path

    return None

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

            # A) 待确认的改名动作
            if conversation_state.pending_action_type == "rename":
                decision = parse_confirmation_decision(question)
                if decision is None:
                    remind = (
                        "当前有待确认的重命名操作。\n"
                        f"{conversation_state.pending_action_preview or ''}\n"
                        "请回复“确认重命名”执行，或回复“取消”。"
                    )
                    print_answer(remind, start_qa)
                    append_memory(memory_buffer, question, remind)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=remind,
                        route="file_action",
                        local_topic="rename_pending",
                        is_content_answer=False,
                    )
                    continue

                if decision == "cancel":
                    canceled = "已取消本次重命名操作。"
                    _clear_pending_action_state()
                    print_answer(canceled, start_qa)
                    append_memory(memory_buffer, question, canceled)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=canceled,
                        route="file_action",
                        local_topic="rename_canceled",
                        is_content_answer=False,
                    )
                    continue

                source_rel = conversation_state.pending_action_source_path
                target_rel = conversation_state.pending_action_target_path
                requested_text = conversation_state.pending_action_requested_text or ""
                if not source_rel or not target_rel:
                    failed = "待执行的重命名状态不完整，已取消。请重新发起改名请求。"
                    _clear_pending_action_state()
                    print_answer(failed, start_qa)
                    append_memory(memory_buffer, question, failed)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=failed,
                        route="file_action",
                        local_topic="rename_failed",
                        is_content_answer=False,
                    )
                    continue

                source_abs = (notes_dir / source_rel).resolve()
                target_abs = (notes_dir / target_rel).resolve()

                if not source_abs.exists():
                    failed = f"原文件不存在或已被移动：{source_rel}。已取消本次重命名。"
                    _clear_pending_action_state()
                    print_answer(failed, start_qa)
                    append_memory(memory_buffer, question, failed)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=failed,
                        route="file_action",
                        local_topic="rename_failed",
                        is_content_answer=False,
                    )
                    continue

                if target_abs.exists() and target_abs != source_abs:
                    failed = f"目标文件已存在：{target_rel}。请换一个名字后重试。"
                    _clear_pending_action_state()
                    print_answer(failed, start_qa)
                    append_memory(memory_buffer, question, failed)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=failed,
                        route="file_action",
                        local_topic="rename_failed",
                        is_content_answer=False,
                    )
                    continue

                before = collect_file_snapshot(source_abs, notes_dir)
                source_abs.rename(target_abs)
                after = collect_file_snapshot(target_abs, notes_dir)
                event_id = change_store.record_rename(
                    notes_dir=notes_dir,
                    before=before,
                    after=after,
                    reason="user_confirmed_rename",
                    requested_text=requested_text,
                    confirmed_text=question,
                )

                apply_repo_state_rename(
                    repo_state,
                    notes_dir=notes_dir,
                    old_rel_path=source_rel,
                    new_rel_path=target_rel,
                )
                if current_focus_file == source_rel:
                    current_focus_file = target_rel
                if conversation_state.last_result_set_items:
                    conversation_state.last_result_set_items = [
                        target_rel if x == source_rel else x
                        for x in conversation_state.last_result_set_items
                    ]

                _clear_pending_action_state()
                success = (
                    "重命名完成。\n"
                    f"- 原路径: {source_rel}\n"
                    f"- 新路径: {target_rel}\n"
                    f"- 变更记录ID: {event_id}\n"
                    f"- 原始SHA256: {before['sha256'][:16]}...\n"
                    f"- 当前SHA256: {after['sha256'][:16]}..."
                )
                print_answer(success, start_qa)
                append_memory(memory_buffer, question, success)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=success,
                    route="file_action",
                    local_topic="rename_done",
                    is_content_answer=True,
                )
                continue

            if conversation_state.pending_action_type == "delete":
                decision = parse_delete_confirmation_decision(question)
                if decision is None:
                    remind = (
                        "当前有待确认的删除操作。\n"
                        f"{conversation_state.pending_action_preview or ''}\n"
                        "请回复“确认删除”执行，或回复“取消”。"
                    )
                    print_answer(remind, start_qa)
                    append_memory(memory_buffer, question, remind)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=remind,
                        route="file_action",
                        local_topic="delete_pending",
                        is_content_answer=False,
                    )
                    continue

                if decision == "cancel":
                    canceled = "已取消本次删除操作。"
                    _clear_pending_action_state()
                    print_answer(canceled, start_qa)
                    append_memory(memory_buffer, question, canceled)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=canceled,
                        route="file_action",
                        local_topic="delete_canceled",
                        is_content_answer=False,
                    )
                    continue

                source_rel = conversation_state.pending_action_source_path
                target_rel = conversation_state.pending_action_target_path
                requested_text = conversation_state.pending_action_requested_text or ""
                if not source_rel or not target_rel:
                    failed = "待执行的删除状态不完整，已取消。请重新发起删除请求。"
                    _clear_pending_action_state()
                    print_answer(failed, start_qa)
                    append_memory(memory_buffer, question, failed)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=failed,
                        route="file_action",
                        local_topic="delete_failed",
                        is_content_answer=False,
                    )
                    continue

                source_abs = (notes_dir / source_rel).resolve()
                target_abs = (notes_dir / target_rel).resolve()

                if not source_abs.exists() or not source_abs.is_file():
                    failed = f"原文件不存在或已被移动：{source_rel}。已取消本次删除。"
                    _clear_pending_action_state()
                    print_answer(failed, start_qa)
                    append_memory(memory_buffer, question, failed)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=failed,
                        route="file_action",
                        local_topic="delete_failed",
                        is_content_answer=False,
                    )
                    continue

                target_abs.parent.mkdir(parents=True, exist_ok=True)
                before = collect_file_snapshot(source_abs, notes_dir)
                source_abs.rename(target_abs)
                after = collect_file_snapshot(target_abs, notes_dir)
                event_id = change_store.record_delete(
                    notes_dir=notes_dir,
                    before=before,
                    after=after,
                    reason="user_confirmed_soft_delete",
                    requested_text=requested_text,
                    confirmed_text=question,
                )

                apply_repo_state_delete(
                    repo_state,
                    notes_dir=notes_dir,
                    old_rel_path=source_rel,
                )
                if current_focus_file == source_rel:
                    current_focus_file = None
                if conversation_state.last_result_set_items:
                    conversation_state.last_result_set_items = [
                        x for x in conversation_state.last_result_set_items if x != source_rel
                    ]
                    if not conversation_state.last_result_set_items:
                        conversation_state.last_result_set_entity_type = None

                _clear_pending_action_state()
                success = (
                    "删除完成（已移动到回收站）。\n"
                    f"- 原路径: {source_rel}\n"
                    f"- 回收站路径: {target_rel}\n"
                    f"- 变更记录ID: {event_id}\n"
                    f"- 文件SHA256: {after['sha256'][:16]}..."
                )
                print_answer(success, start_qa)
                append_memory(memory_buffer, question, success)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=success,
                    route="file_action",
                    local_topic="delete_done",
                    is_content_answer=True,
                )
                continue

            if is_change_history_query(question):
                limit = extract_history_limit(question, default=20, max_limit=100)
                events = change_store.list_recent_events(notes_dir=notes_dir, limit=limit)
                if not events:
                    history_answer = "当前还没有文件变更记录。"
                else:
                    action_map = {
                        "rename": "重命名",
                        "delete": "删除(软删除)",
                    }
                    lines = [f"最近 {len(events)} 条文件变更记录（按时间倒序）："]
                    for i, evt in enumerate(events, start=1):
                        created_at = str(evt.get("created_at", "")).replace("T", " ").replace("+00:00", "Z")
                        event_type = action_map.get(evt.get("event_type", ""), evt.get("event_type", "unknown"))
                        lines.append(
                            f"{i}. [{evt['event_id']}] {event_type}: "
                            f"{evt['before_path']} -> {evt['after_path']} "
                            f"(SHA {evt['before_sha256'][:8]} -> {evt['after_sha256'][:8]} | {created_at})"
                        )
                    history_answer = "\n".join(lines)

                print_answer(history_answer, start_qa)
                append_memory(memory_buffer, question, history_answer)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=history_answer,
                    route="file_action",
                    local_topic="change_history",
                    is_content_answer=False,
                )
                continue

            # A.1) 重命名历史查询
            if is_rename_history_query(question):
                recent_events = change_store.list_recent_renames(notes_dir=notes_dir, limit=20)
                if not recent_events:
                    history_answer = "当前还没有重命名记录。"
                else:
                    lines = ["最近重命名记录如下（按时间倒序）："]
                    for i, evt in enumerate(recent_events, start=1):
                        created_at = str(evt.get("created_at", "")).replace("T", " ").replace("+00:00", "Z")
                        lines.append(
                            f"{i}. [{evt['event_id']}] {evt['before_path']} -> {evt['after_path']} "
                            f"(SHA {evt['before_sha256'][:8]} -> {evt['after_sha256'][:8]} | {created_at})"
                        )
                    history_answer = "\n".join(lines)

                print_answer(history_answer, start_qa)
                append_memory(memory_buffer, question, history_answer)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=history_answer,
                    route="file_action",
                    local_topic="rename_history",
                    is_content_answer=False,
                )
                continue

            # B) 新改名请求：先预览，再确认
            if is_rename_request(question):
                source_hint = resolve_source_file(
                    question=question,
                    current_focus_file=current_focus_file,
                    last_result_set_items=conversation_state.last_result_set_items,
                )
                source_rel = _find_repo_path_by_reference(source_hint, list(repo_state.paths))
                new_name_candidate = extract_new_name_candidate(question)

                if not source_rel:
                    tip = "我没定位到要改名的文件。请带上完整文件名再说一次。"
                    print_answer(tip, start_qa)
                    append_memory(memory_buffer, question, tip)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=tip,
                        route="file_action",
                        local_topic="rename_pending",
                        is_content_answer=False,
                    )
                    continue

                if not new_name_candidate:
                    tip = f"我识别到了目标文件 `{source_rel}`，但没识别到新名称。请明确说“改成 xxx”。"
                    print_answer(tip, start_qa)
                    append_memory(memory_buffer, question, tip)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=tip,
                        route="file_action",
                        local_topic="rename_pending",
                        is_content_answer=False,
                    )
                    continue

                target_rel = normalize_target_filename(new_name_candidate, source_rel)
                if not target_rel:
                    tip = "新文件名不合法或格式不支持。请换一个不含特殊字符的名字。"
                    print_answer(tip, start_qa)
                    append_memory(memory_buffer, question, tip)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=tip,
                        route="file_action",
                        local_topic="rename_pending",
                        is_content_answer=False,
                    )
                    continue

                if target_rel == source_rel:
                    tip = f"文件名已经是 `{Path(source_rel).name}`，无需修改。"
                    print_answer(tip, start_qa)
                    append_memory(memory_buffer, question, tip)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=tip,
                        route="file_action",
                        local_topic="rename_pending",
                        is_content_answer=False,
                    )
                    continue

                preview_text, payload = build_rename_preview(
                    notes_dir=notes_dir,
                    source_rel_path=source_rel,
                    target_rel_path=target_rel,
                )
                if not preview_text or not payload:
                    tip = "重命名预检查失败：源文件不存在，或目标文件已存在。"
                    print_answer(tip, start_qa)
                    append_memory(memory_buffer, question, tip)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=tip,
                        route="file_action",
                        local_topic="rename_pending",
                        is_content_answer=False,
                    )
                    continue

                conversation_state.pending_action_type = "rename"
                conversation_state.pending_action_source_path = payload["source_rel_path"]
                conversation_state.pending_action_target_path = payload["target_rel_path"]
                conversation_state.pending_action_requested_text = question
                conversation_state.pending_action_preview = preview_text

                print_answer(preview_text, start_qa)
                append_memory(memory_buffer, question, preview_text)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=preview_text,
                    route="file_action",
                    local_topic="rename_preview",
                    is_content_answer=False,
                )
                continue

            if is_delete_request(question):
                source_hint = resolve_delete_source_file(
                    question=question,
                    current_focus_file=current_focus_file,
                    last_result_set_items=conversation_state.last_result_set_items,
                )
                source_rel = _find_repo_path_by_reference(source_hint, list(repo_state.paths))
                if not source_rel:
                    tip = "我没定位到要删除的文件，请带上完整文件名再说一次。"
                    print_answer(tip, start_qa)
                    append_memory(memory_buffer, question, tip)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=tip,
                        route="file_action",
                        local_topic="delete_pending",
                        is_content_answer=False,
                    )
                    continue

                preview_text, payload = build_delete_preview(
                    notes_dir=notes_dir,
                    source_rel_path=source_rel,
                )
                if not preview_text or not payload:
                    tip = "删除预检查失败：源文件不存在，或文件类型不支持。"
                    print_answer(tip, start_qa)
                    append_memory(memory_buffer, question, tip)
                    conversation_state = update_state_after_local_answer(
                        conversation_state,
                        question=question,
                        answer=tip,
                        route="file_action",
                        local_topic="delete_failed",
                        is_content_answer=False,
                    )
                    continue

                conversation_state.pending_action_type = "delete"
                conversation_state.pending_action_source_path = payload["source_rel_path"]
                conversation_state.pending_action_target_path = payload["target_rel_path"]
                conversation_state.pending_action_requested_text = question
                conversation_state.pending_action_preview = preview_text

                print_answer(preview_text, start_qa)
                append_memory(memory_buffer, question, preview_text)
                conversation_state = update_state_after_local_answer(
                    conversation_state,
                    question=question,
                    answer=preview_text,
                    route="file_action",
                    local_topic="delete_preview",
                    is_content_answer=False,
                )
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
