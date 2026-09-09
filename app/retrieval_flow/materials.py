from __future__ import annotations

from ai.prompt_builder import build_final_prompt
from app.retrieval_flow.query import (
    filter_reused_indices_for_question,
    should_reuse_previous_results,
)
from app.chat_text.core import (
    build_timeline_evidence_text,
    extract_timeline_evidence_from_chunks,
    needs_timeline_evidence,
    redact_sensitive_text,
)
from app.chat_text.file_lookup import (
    looks_like_all_items_file_set_content_question,
    looks_like_bare_content_question,
)
from retrieval.attribute_evidence import classify_requested_attribute_kind
from retrieval.search_engine import (
    build_context_text,
    build_inventory_candidates_text,
    perform_retrieval,
)
from retrieval.search_context import build_context_source_candidates


def build_retrieval_materials(
    *,
    question: str,
    search_query: str,
    context_anchor: str,
    flags: dict,
    repo_state,
    model_emb,
    logger,
    current_focus_file,
    last_relevant_indices=None,
    event=None,
    allowed_paths=None,
    scope_label: str | None = None,
    selected_source_files: list[str] | None = None,
    content_target: str | None = None,
    answer_entity_followup: bool = False,
    answer_entity_items: list[str] | None = None,
    include_source_ids: bool = False,
):
    inventory_candidates_text = (
        build_inventory_candidates_text(question, repo_state, flags["inventory_target_type"])
        if flags["is_inventory_query"]
        else ""
    )

    context_text = ""
    timeline_evidence_text = ""
    relevant_indices = []
    context_source_candidates = ()

    if not flags["skip_retrieval"]:
        event_name = getattr(event, "name", None)
        requested_attribute_kind = classify_requested_attribute_kind(question)
        file_attribute_followup = bool(
            event_name == "result_set_followup"
            and requested_attribute_kind is not None
        )
        ensure_file_set_content_coverage = bool(
            allowed_paths is not None
            and (
                answer_entity_followup
                or file_attribute_followup
                or event_name == "synthesis_request"
                or (
                    event_name == "result_set_followup"
                    and (
                        looks_like_all_items_file_set_content_question(question)
                        or looks_like_bare_content_question(question)
                    )
                )
            )
        )
        effective_allowed_paths = allowed_paths
        if event_name == "selected_candidate_followup" and selected_source_files:
            selected_path_set = {str(path or "").strip() for path in selected_source_files if str(path or "").strip()}
            if effective_allowed_paths is None:
                effective_allowed_paths = selected_path_set
            else:
                effective_allowed_paths = selected_path_set.intersection(
                    {str(path or "").strip() for path in effective_allowed_paths if str(path or "").strip()}
                )
            logger.info(f"🎯 [选择焦点范围] 限定为 {len(effective_allowed_paths)} 个来源文件")

        reuse_previous_results = should_reuse_previous_results(question, event, last_relevant_indices)

        if reuse_previous_results:
            logger.info("♻️ [追问复用] 使用上一轮检索结果，并按当前问题二次过滤")
            relevant_indices = filter_reused_indices_for_question(
                question=question,
                candidate_indices=last_relevant_indices,
                repo_state=repo_state,
                logger=logger,
            )
            if effective_allowed_paths is not None:
                chunk_paths = list(getattr(repo_state, "chunk_paths", []) or [])
                allowed_path_set = {
                    str(path or "").strip()
                    for path in effective_allowed_paths
                    if str(path or "").strip()
                }
                relevant_indices = [
                    idx
                    for idx in relevant_indices
                    if idx < len(chunk_paths) and str(chunk_paths[idx] or "").strip() in allowed_path_set
                ]
        else:
            retrieval = perform_retrieval(
                question,
                search_query,
                repo_state,
                model_emb,
                logger,
                current_focus_file,
                context_anchor=context_anchor,
                allowed_paths=effective_allowed_paths,
                scope_label=scope_label,
                task_mode=getattr(event, "name", None),
                content_target=content_target,
                ensure_allowed_path_coverage=ensure_file_set_content_coverage,
                answer_entity_items=(
                    answer_entity_items if answer_entity_followup else None
                ),
                requested_attribute=(
                    question
                    if answer_entity_followup
                    else search_query
                    if file_attribute_followup
                    else None
                ),
            )
            current_focus_file = retrieval["current_focus_file"]
            relevant_indices = retrieval["relevant_indices"]
            attribute_evidence_indices = retrieval.get(
                "attribute_evidence_indices",
                [],
            )

        per_file_evidence_limit = (
            6
            if answer_entity_followup
            else
            2
            if (
                ensure_file_set_content_coverage
                and event_name == "synthesis_request"
                and bool(content_target)
            )
            else 3
        )
        if per_file_evidence_limit == 2:
            logger.info("📚 [集合分类证据预算] 每个活动文件最多保留 2 个片段")
        elif answer_entity_followup:
            logger.info("🧩 [实体属性上下文预算] 总片段限制为 14，每文件最多 6 个")
        context_text = build_context_text(
            relevant_indices,
            repo_state,
            logger,
            per_file_limit=per_file_evidence_limit,
            total_limit=(14 if answer_entity_followup else None),
            include_source_ids=include_source_ids,
        )
        context_source_candidates = build_context_source_candidates(
            relevant_indices,
            repo_state,
            per_file_limit=per_file_evidence_limit,
            total_limit=(14 if answer_entity_followup else None),
        )

        if needs_timeline_evidence(question):
            timeline_items = extract_timeline_evidence_from_chunks(
                relevant_indices,
                repo_state,
            )
            timeline_evidence_text = build_timeline_evidence_text(timeline_items)

    return {
        "inventory_candidates_text": inventory_candidates_text,
        "context_text": context_text,
        "timeline_evidence_text": timeline_evidence_text,
        "current_focus_file": current_focus_file,
        "relevant_indices": relevant_indices,
        "context_source_candidates": context_source_candidates,
    }


def build_safe_final_prompt(
    *,
    memory_buffer: list[str],
    current_focus_file,
    inventory_candidates_text: str,
    context_text: str,
    timeline_evidence_text: str,
    question: str,
    event_name: str | None = None,
    result_set_items: list[str] | None = None,
    result_set_entity_type: str | None = None,
    answer_entity_followup: bool = False,
    selected_candidate: str | None = None,
    selected_source_files: list[str] | None = None,
    comparison_source_files: list[str] | None = None,
) -> str:
    safe_memory_buffer = [redact_sensitive_text(x) for x in memory_buffer]
    safe_inventory_candidates_text = redact_sensitive_text(inventory_candidates_text)
    safe_context_text = redact_sensitive_text(timeline_evidence_text + context_text)
    safe_question = redact_sensitive_text(question)
    safe_result_set_items = [redact_sensitive_text(x) for x in (result_set_items or [])]

    constrained_context_text = safe_context_text
    constrained_question = safe_question
    is_file_set_content_operation = bool(
        event_name == "result_set_followup"
        and (
            looks_like_all_items_file_set_content_question(question)
            or looks_like_bare_content_question(question)
            or classify_requested_attribute_kind(question) is not None
        )
    )

    if safe_result_set_items and answer_entity_followup:
        entity_label = redact_sensitive_text(result_set_entity_type or "对象")
        answer_entity_block = (
            "【上一轮回答实体集合】\n"
            + f"实体类型：{entity_label}\n"
            + "\n".join(f"- {item}" for item in safe_result_set_items[:20])
            + "\n\n"
            "【回答实体语义范围约束】\n"
            "当前问题是在询问上述实体集合的属性、条件或分组。"
            "这些条目只用于承接上一轮回答语义，不代表可以按序号安全定位来源。"
            "请只围绕上述实体回答，不新增集合外实体；按参考片段逐项核对。"
            "只有参考片段明确给出否定、排除或完整适用范围时，才可断言来源没有、未提及、"
            "不存在或不适用；本轮有限片段没有命中时，只能说明“当前检索到的证据中暂未找到明确说明”，"
            "不得把检索缺口写成来源事实。标题、主要对象、常见群体或只针对某群体给出的统计，"
            "都不能自动推出其他群体被排除；必须有“仅限、不得、不适用、不包括”等明确边界证据。\n\n"
            "宽泛集合或群体名称也不能自动证明任一特定子群体已被覆盖。\n\n"
        )
        constrained_context_text = answer_entity_block + constrained_context_text

    if safe_result_set_items and event_name in {
        "result_set_followup",
        "result_set_expansion_followup",
        "structured_request",
        "synthesis_request",
    }:
        if is_file_set_content_operation:
            result_set_block = (
                "【活动文件结果集】\n"
                + "\n".join(f"{index}. {item}" for index, item in enumerate(safe_result_set_items[:20], 1))
                + "\n\n"
                "【文件结果集内容操作约束】\n"
                "当前问题以整个活动文件结果集为对象。请严格按上述顺序覆盖每个文件，不能漏项，"
                "也不能新增集合外文件。每项只使用该文件的参考片段；证据不足时在对应项明确说明。\n\n"
            )
            requested_attribute_kind = classify_requested_attribute_kind(question)
            if requested_attribute_kind == "source":
                result_set_block += (
                    "【来源属性判断约束】\n"
                    "只依据正文中可核对的发布、制定、编制、署名或来源信息判断；"
                    "不能根据文件名、标题中的“指南”“规范”等正式措辞猜测来源性质。"
                    "企业、社会组织或其他机构来源应按证据如实说明，不等同于政府官方发布。"
                    "回答必须停在来源或主体这一事实层级，不得据此推导材料的正式类别或效力。"
                    "没有足够证据的文件必须逐项写明无法确认。\n\n"
                )
            elif requested_attribute_kind == "document_property":
                result_set_block += (
                    "【材料性质判断约束】\n"
                    "当前问题判断的是材料本身的类别、性质或效力，不是来源主体。"
                    "来源/provenance 证据与材料性质证据是两个独立维度：发布、印发、"
                    "委托制定、编制、署名或由某机构提供，只能支持相应的来源或主体事实；"
                    "即使主体属于官方机构，也不能据此自动判定材料具有正式标准、正式规范"
                    "或其他正式文件属性。标题或文件名中的类别措辞只能说明其标题或自称，"
                    "不能单独证明法定、正式或强制属性。"
                    "只有参考片段直接说明所问材料性质时才可确认；若片段只支持来源主体，"
                    "应先如实说明该来源事实，再明确写明材料性质无法确认。"
                    "必须按活动文件结果集逐项作答；证据不足不得用来源事实补足结论。\n\n"
                )
        elif event_name == "result_set_followup":
            result_set_block = (
                "【上一轮候选集合】\n"
                + "\n".join(f"- {item}" for item in safe_result_set_items[:20])
                + "\n\n"
                "【结果集追问约束】\n"
                "当前问题是在上一轮候选集合基础上的进一步筛选。\n"
                "你只能在上述候选项中进行判断，不得新增集合外实体。\n"
                "若证据不足，可回答“无法确定”，不要扩展候选集合。\n\n"
            )
        elif event_name == "result_set_expansion_followup":
            result_set_block = (
                "【已知候选集合】\n"
                + "\n".join(f"- {item}" for item in safe_result_set_items[:20])
                + "\n\n"
                "【结果集扩展约束】\n"
                "这是在已知候选基础上的补充追问，可以新增集合外实体。\n"
                "新增项必须有参考片段证据，并避免重复已知候选。\n"
                "若没有新增，请明确说明“没有识别出新的实体”。\n\n"
            )
        elif event_name == "structured_request":
            result_set_block = (
                "【上一轮候选集合】\n"
                + "\n".join(f"- {item}" for item in safe_result_set_items[:20])
                + "\n\n"
                "【结构化整理约束】\n"
                "当前问题是在上一轮候选集合基础上做结构化整理。\n"
                "请按候选集合逐项输出，不能漏项；若某项字段缺失，请写“未知”或“未明确”。\n"
                "不要新增集合外实体。\n\n"
            )
        else:
            result_set_block = (
                "【当前活动文件集合】\n"
                + "\n".join(f"- {item}" for item in safe_result_set_items[:20])
                + "\n\n"
                "【集合范围约束】\n"
                "当前问题承接上一轮集合级回答。请继续以整个活动文件集合为语义范围，"
                "只使用该集合内的参考片段展开，不得收缩为单个文件或扩展到集合外文件。\n\n"
            )
        constrained_context_text = result_set_block + constrained_context_text

    return build_final_prompt(
        memory_buffer=safe_memory_buffer,
        current_focus_file=current_focus_file,
        inventory_candidates_text=safe_inventory_candidates_text,
        context_text=constrained_context_text,
        question=safe_question,
        event_name=event_name,
        result_set_items=safe_result_set_items,
        selected_candidate=redact_sensitive_text(selected_candidate or ""),
        selected_source_files=[redact_sensitive_text(x) for x in (selected_source_files or [])],
        comparison_source_files=[redact_sensitive_text(x) for x in (comparison_source_files or [])],
    )
