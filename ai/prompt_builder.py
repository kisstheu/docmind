from __future__ import annotations


def build_focus_injection(current_focus_file: str | None) -> str:
    if not current_focus_file:
        return ""
    return (
        f"【当前焦点】当前对话优先围绕文件《{current_focus_file}》展开。"
        f"如果用户没有明确切换对象，请优先参考该文件及其相关片段，"
        f"不要随意扩展到其他无关文件或人物。\n"
    )


def build_final_prompt(memory_buffer, current_focus_file, inventory_candidates_text, context_text, question):
    focus_injection = build_focus_injection(current_focus_file)
    return (
        f"【近期聊天上下文】:\n{chr(10).join(memory_buffer[-4:])}\n\n"
        f"{focus_injection}"
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
        f"三、表达方式\n"
        f"回答要自然、直接、清楚，不要写成客服话术，也不要故作犀利。"
        f"除非用户明确要求“有哪些”“多少”“列出来”，否则尽量不用列表。"
        f"如果是在评价某个人，只能评价参考片段里能够明确支撑的那部分表现，"
        f"不要把一次互动上升为完整人格结论。\n\n"
        f"四、信息不足时\n"
        f"当证据不够时，请明确指出“目前只能看到这件事里的表现”或“现有材料不足以下结论”。"
        f"宁可收一点，也不要硬猜。\n"
    )
