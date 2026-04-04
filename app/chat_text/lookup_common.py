from __future__ import annotations

import re

DIRECT_LOOKUP_MARKERS = (
    "哪个",
    "哪家",
    "哪位",
    "哪些",
    "谁",
    "对应",
    "分别",
    "关联",
    "匹配",
    "在哪里",
    "在哪",
    "是什么",
    "是啥",
    "有哪",
)
DIRECT_LOOKUP_FOLLOWUP_MARKERS = (
    "呢",
    "这个",
    "那个",
    "还有",
    "继续",
    "再来",
    "只看",
    "仅看",
)
DIRECT_LOOKUP_ANALYSIS_MARKERS = (
    "为什么",
    "怎么",
    "如何",
    "原因",
    "总结",
    "分析",
    "复盘",
    "比较",
    "区别",
    "优缺点",
    "建议",
    "判断",
)
DIRECT_LOOKUP_NON_LOOKUP_MARKERS = (
    "职责",
    "要求",
    "工作内容",
    "原理",
    "流程",
    "步骤",
    "详细介绍",
    "详细说明",
    "怎么做",
    "如何做",
)
DIRECT_LOOKUP_STOP_TERMS = {
    "请问",
    "帮我",
    "麻烦",
    "看看",
    "看下",
    "一下",
    "以及",
    "还有",
    "这个",
    "那个",
    "就是",
    "然后",
    "再",
    "是",
    "的",
}
DIRECT_LOOKUP_SELECTOR_PATTERN = re.compile(
    r"(?:\d{1,4}\s*[-–—~]\s*\d{1,4}(?:\s*[a-zA-Z%]+)?)|(?:\d{4}-\d{2}-\d{2})"
)
DIRECT_LOOKUP_STRUCTURED_LINE_PATTERN = re.compile(r"[:：|/·（）()【】\[\]#]")
RANGE_SIGNATURE_PATTERN = re.compile(r"(\d{1,4})\s*[-–—~]\s*(\d{1,4})(?:\s*[a-zA-Z%]+)?")
DIRECT_LOOKUP_GENERIC_ANCHOR_TERMS = {
    "信息",
    "内容",
    "线索",
    "相关",
    "对应",
    "匹配",
    "结果",
    "详情",
    "说明",
    "数据",
    "记录",
    "文档",
    "文件",
    "文本",
    "对象",
    "实体",
}
ROLE_NAME_QUERY_MARKERS = ("谁", "哪位", "名字", "姓名", "叫什么")
ROLE_NAME_FOLLOWUP_MARKERS = ("呢", "吗", "?", "？")
ROLE_NON_NAME_QUERY_MARKERS = (
    "地点",
    "地址",
    "时间",
    "日期",
    "价格",
    "数量",
    "内容",
    "描述",
    "流程",
    "步骤",
    "原因",
    "背景",
)
ROLE_TERM_HINTS = (
    "hr",
    "hrbp",
    "pm",
    "po",
    "qa",
    "sre",
    "ops",
    "it",
    "rd",
    "ui",
    "ux",
    "owner",
    "lead",
    "manager",
    "director",
    "maintainer",
    "contact",
    "author",
    "editor",
    "负责人",
    "联系人",
    "hr",
    "经理",
    "主管",
    "总监",
    "专员",
    "顾问",
    "老师",
    "秘书",
    "主任",
    "委员",
    "代表",
)
GENERIC_ROLE_NORMS = {
    "工程师",
    "开发",
    "职位",
    "岗位",
    "hr",
    "hrbp",
    "经理",
    "主管",
    "总监",
    "专员",
    "顾问",
    "老师",
    "负责人",
    "联系人",
}
COMPANY_QUERY_MARKERS = ("公司", "机构", "组织", "单位", "团队", "部门", "平台", "学校", "医院", "实验室", "研究院")
COMPANY_BAN_TERMS = (
    "描述",
    "内容",
    "条目",
    "线索",
    "来源",
    "证据",
)
COMPANY_LINE_PATTERNS = (
    re.compile(
        r"([A-Za-z0-9\u4e00-\u9fa5·（）()\-]{2,}"
        r"(?:有限责任公司|股份有限公司|有限公司|集团|大学|学院|研究院|研究所|实验室|医院|银行|协会|事务所|工作室))",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^([A-Za-z0-9\u4e00-\u9fa5·（）()\-]{2,24})\s*·\s*"
        r"(?:联系人|负责人|.*经理|.*主管|.*总监|.*专员|.*顾问|.*老师)$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^([A-Za-z0-9\u4e00-\u9fa5·（）()\-]{2,24})\s*·\s*"
        r"(?:hrbp|hr|recruiter|talent|招聘(?:经理|专员|顾问|官)?|猎头|人才顾问)",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"([A-Za-z0-9&.\- ]{2,40}\b(?:inc|ltd|llc|corp|group|studio|lab|institute|university|bank)\b)",
        flags=re.IGNORECASE,
    ),
)
PERSON_NAME_PATTERNS = (
    re.compile(r"([\u4e00-\u9fa5]{1,4}(?:先生|女士|老师))"),
    re.compile(r"\b([A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20})\b"),
)
MAPPING_FOLLOWUP_BLOCK_TERMS = (
    "原理",
    "流程",
    "步骤",
    "为什么",
    "怎么做",
    "如何做",
    "建议",
    "复盘",
    "总结",
    "分析",
)
DETAIL_LINE_HINT_TERMS = (
    "负责",
    "要求",
    "步骤",
    "流程",
    "方法",
    "说明",
    "描述",
    "背景",
    "原因",
    "使用",
    "通过",
    "以及",
    "包括",
    "例如",
    "比如",
    "用于",
    "实现",
)


def _looks_like_detail_line(line: str) -> bool:
    raw = (line or "").strip()
    if not raw:
        return False
    norm = _normalize_lookup_token(raw)
    if not norm:
        return False
    if re.match(r"^[（(]?\d+[)）.、]", raw):
        return True
    if any(term in norm for term in DETAIL_LINE_HINT_TERMS):
        return True
    if len(raw) > 72:
        return True
    if sum(1 for ch in raw if ch in "，。；;：:") >= 2:
        return True
    return False


def _is_plausible_person_name(name: str, evidence_line: str) -> bool:
    n = (name or "").strip()
    if not n:
        return False
    if re.search(r"[\u4e00-\u9fa5]{1,4}(?:先生|女士|老师)$", n):
        return True
    if re.fullmatch(r"[A-Z][a-z]{1,20}\s+[A-Z][a-z]{1,20}", n):
        line = (evidence_line or "").strip()
        if not line:
            return False
        escaped_name = re.escape(n)
        role_marker = r"(?:hr|hrbp|recruiter|contact|mr|ms|mrs|dr|\u8054\u7cfb\u4eba|\u4eba\u4e8b|\u62db\u8058\u8d1f\u8d23\u4eba)"
        if re.search(rf"{role_marker}\s*[:：\-]?\s*{escaped_name}", line, flags=re.IGNORECASE):
            return True
        if re.search(rf"{escaped_name}\s*[:：\-]?\s*{role_marker}", line, flags=re.IGNORECASE):
            return True
        if re.search(
            rf"{escaped_name}.*(?:wechat|phone|mobile|email|tel|\u5fae\u4fe1|\u7535\u8bdd|\u624b\u673a|\u90ae\u7bb1)",
            line,
            flags=re.IGNORECASE,
        ):
            return True
        return False
    return False


def _looks_like_heading_line(line: str) -> bool:
    raw = (line or "").strip()
    if not raw:
        return False
    if len(raw) > 48:
        return False
    if re.match(r"^[（(]?\d+[)）.、]", raw):
        return False
    if any(ch in raw for ch in "，。；;：:"):
        return False
    if _looks_like_detail_line(raw):
        return False
    return True


def _normalize_lookup_token(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "", (text or "").lower()).strip()


def _is_ascii_token(token: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9]+", token or ""))


def _term_matches_line(term_norm: str, raw_line_lower: str, line_norm: str) -> bool:
    if not term_norm:
        return False

    if _is_ascii_token(term_norm):
        if term_norm == "hr":
            tokens = re.findall(r"[a-z0-9]+", raw_line_lower)
            return any(tok == "hr" or tok.startswith("hr") for tok in tokens)

        # 英文/数字词使用词边界，避免 hr 命中 chroma 这类子串。
        pattern = rf"(?<![a-z0-9]){re.escape(term_norm)}(?![a-z0-9])"
        return bool(re.search(pattern, raw_line_lower))

    return term_norm in line_norm


def _term_matches_text(term_norm: str, text_lower: str, text_norm: str) -> bool:
    if not term_norm:
        return False

    if _is_ascii_token(term_norm):
        if term_norm == "hr":
            tokens = re.findall(r"[a-z0-9]+", text_lower)
            return any(tok == "hr" or tok.startswith("hr") for tok in tokens)
        pattern = rf"(?<![a-z0-9]){re.escape(term_norm)}(?![a-z0-9])"
        return bool(re.search(pattern, text_lower))

    return term_norm in text_norm


def _extract_range_signatures(text: str) -> set[str]:
    signatures: set[str] = set()
    for lo, hi in RANGE_SIGNATURE_PATTERN.findall(text or ""):
        signatures.add(f"{int(lo)}-{int(hi)}")
    return signatures


def _extract_selector_signatures(text: str) -> set[str]:
    signatures = _extract_range_signatures(text or "")
    for m in re.findall("(\\d{1,2})\\s*\\u85aa", text or "", flags=re.IGNORECASE):
        signatures.add(f"salaryx{int(m)}")
    return signatures


def _selector_constraints_satisfied(file_signatures: set[str], required_signatures: set[str]) -> bool:
    if not required_signatures:
        return True
    if not file_signatures:
        # When OCR misses selector tokens in a file, avoid hard false-negative.
        return True

    required_salary = {s for s in required_signatures if s.startswith("salaryx")}
    file_salary = {s for s in file_signatures if s.startswith("salaryx")}
    if required_salary and file_salary and not (required_salary & file_salary):
        return False

    required_range = {
        s for s in required_signatures if re.fullmatch(r"\d{1,4}-\d{1,4}", s)
    }
    file_range = {s for s in file_signatures if re.fullmatch(r"\d{1,4}-\d{1,4}", s)}
    if required_range and file_range and not (required_range & file_range):
        return False

    required_other = required_signatures - required_salary - required_range
    if required_other and not (required_other & file_signatures):
        return False

    return True


__all__ = [name for name in globals() if not name.startswith("__")]

