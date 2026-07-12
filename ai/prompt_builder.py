from __future__ import annotations


def build_focus_injection(current_focus_file: str | None) -> str:
    if not current_focus_file:
        return ""
    return (
        f"【当前焦点】当前对话优先围绕文件《{current_focus_file}》展开。"
        f"如果用户没有明确切换对象，请优先参考该文件及其相关片段，"
        f"不要随意扩展到其他无关文件或人物。\n"
    )


def build_final_prompt(
    memory_buffer,
    current_focus_file,
    inventory_candidates_text,
    context_text,
    question,
    event_name=None,
    result_set_items=None,
    selected_candidate=None,
    selected_source_files=None,
):
    focus_injection = build_focus_injection(current_focus_file)

    result_set_injection = ""
    if result_set_items and event_name in {"result_set_followup", "result_set_expansion_followup"}:
        if event_name == "result_set_followup":
            result_set_injection = (
                "【上一轮候选集合】\n"
                + "\n".join(f"- {x}" for x in result_set_items[:20])
                + "\n\n"
                + "【结果集追问约束】\n"
                + "当前问题是在上一轮候选集合基础上的进一步筛选。"
                + "只能在上述候选项中做判断，不得新增集合外实体。"
                + "若证据不足，可明确写“无法确定”，不要跳出候选集合。\n\n"
            )
        else:
            result_set_injection = (
                "【已知候选集合】\n"
                + "\n".join(f"- {x}" for x in result_set_items[:20])
                + "\n\n"
                + "【结果集扩展约束】\n"
                + "当前问题是在已知候选基础上的补充查找，可以补充新的实体。"
                + "新增实体必须有参考片段证据，且不要重复已知候选。"
                + "若无新增，请直接说明“没有识别出新的实体”。\n\n"
            )

    entity_name_constraint = (
        "【实体名称约束】\n"
        "当列举公司、人物、项目等实体时，只允许输出材料中能明确确认的正式名称。\n"
        "如果材料里只有泛指、代称、简称、地名+公司、某公司、该公司等不完整名称，"
        "不要把它们当作独立正式实体列出。\n"
        "若无法确定完整名称，请直接忽略该项，不要补写、造词或泛化。\n\n"
    )

    task_constraint = ""
    if event_name == "synthesis_request":
        task_constraint = (
            "【集合归纳任务】\n"
            "这是跨多个证据的归纳，不是定位或片段列举。请从参考片段归纳用户所问属性，"
            "按证据自然形成的维度组织；每项标注来源文件。"
            "标题、联系人、作者、文件名等身份信息不能冒充用户所问的要求、条件、能力或属性。"
            "证据覆盖不足时明确范围，不得补写。\n\n"
        )
    elif event_name == "decision_request":
        task_constraint = (
            "【比较与决策任务】\n"
            "这是检索、比较再决策的综合任务。先用用户明确给出的条件逐项比较候选，再给结论；"
            "不得只返回证据片段。若用户要求推荐固定数量，必须严格遵守数量。"
            "必须严格区分：方向匹配、用户明确提供的能力、尚未被用户信息证明的要求、明显风险。"
            "用户说研究过某方向，只能视为方向匹配，不能自动等同于项目实施、生产落地、交付或上线经验。"
            "“已有明确能力”只能复述【用户最新提问】中明确提供的事实，不得从候选要求反向补全。"
            "候选材料提出、但用户没有明确证明满足的要求，必须放入“未被证据证明的要求”或风险。"
            "推荐成功时必须逐行使用以下字段，字段名不得改写：\n"
            "推荐结论：<简短结论，不在这里写候选名>\n"
            "推荐对象：<且仅写一个候选实体>\n"
            "推荐理由：<为何相对更合适>\n"
            "方向匹配：<只写方向层面的对应>\n"
            "已有明确能力：<只写用户原话能证明的事实>\n"
            "未被证据证明的要求：<待确认项>\n"
            "明显差距或风险：<差距与风险>\n"
            "来源文件：<且仅写被推荐对象的来源文件>\n"
            "若没有可靠候选，必须只写“推荐结论：暂无足够匹配的候选”，不得输出推荐对象。\n\n"
        )
    elif event_name == "selected_candidate_followup" and selected_candidate:
        source_lines = "\n".join(f"- {item}" for item in (selected_source_files or [])[:10])
        task_constraint = (
            "【已选对象追问】\n"
            f"上一轮已选对象：{selected_candidate}\n"
            + (f"上一轮来源文件：\n{source_lines}\n" if source_lines else "")
            + "本轮只围绕该对象详细分析，不要重新扩展为所有候选；证据不足处明确说明。\n\n"
        )

    return (
        f"【近期聊天上下文】:\n{chr(10).join(memory_buffer[-4:])}\n\n"
        f"{focus_injection}"
        f"{result_set_injection}"
        f"{task_constraint}"
        f"{inventory_candidates_text}"
        f"{context_text}"
        f"【用户最新提问】\n{question}\n\n"
        f"【本轮回答规则】\n"
        f"一、回答依据\n"
        f"你的判断必须优先建立在【参考片段】上。"
        f"如果参考片段能直接回答，就直接回答；"
        f"如果只能支持局部结论，就只回答局部；"
        f"如果支持不了，就明确说信息不足，不要补全。\n\n"
        f"二、实体隔离\n"
        f"不同时间、人物、公司、项目要严格分开。"
        f"名字相似、称呼相似、同姓、简称相似，都不能自动视为同一个对象。"
        f"只有参考片段里出现了明确证据，才允许合并判断。\n\n"
        f"{entity_name_constraint}"
        f"三、表达方式\n"
        f"回答要自然、直接、清楚，不要写成客服话术，也不要故作犀利。"
        f"除非用户明确要求“有哪些”“多少”“列出来”，否则尽量不用列表。"
        f"如果是在评价某个人，只能评价参考片段里能够明确支撑的那部分表现，"
        f"不要把一次互动上升为完整人格结论。\n\n"
        f"四、信息不足时\n"
        f"当证据不够时，请明确指出“目前只能看到这件事里的表现”或“现有材料不足以下结论”。"
        f"宁可收一点，也不要硬猜。\n"
    )
