from __future__ import annotations

import re
from pathlib import Path


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


def redact_sensitive_text(text: str) -> str:
    t = text or ""
    t = re.sub(r"\b\d{17}[\dXx]\b", "[身份证号已脱敏]", t)
    t = re.sub(r"\b1[3-9]\d{9}\b", "[手机号已脱敏]", t)
    t = re.sub(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", "[邮箱已脱敏]", t)
    t = re.sub(r"\b\d{16,19}\b", "[长数字已脱敏]", t)
    return t


def strip_structured_request_words(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    t = re.sub(r"^(给我|帮我|请你|麻烦你|我想|我先|先)\s*", "", t)
    t = re.sub(r"(吧|吗|呢|呀|啊)$", "", t)
    t = re.sub(r"(时间线|时间顺序|整理一下|梳理一下|分析一下|总结一下)", " ", t)

    if t in {"更详细的", "详细的", "详细点", "更详细", "详细一些"}:
        return ""

    return re.sub(r"\s+", " ", t).strip()


def build_clean_merged_query(event_merged_query: str, current_question: str) -> str:
    parent = strip_structured_request_words(event_merged_query)
    current = strip_structured_request_words(current_question)

    if not parent and not current:
        return (current_question or "").strip()
    if not parent:
        return current or (current_question or "").strip()
    if not current:
        return parent

    return re.sub(r"\s+", " ", f"{parent} {current}".strip())


QUERY_FILLERS = {
    "帮我",
    "帮忙",
    "请",
    "麻烦",
    "看下",
    "看看",
    "分析下",
    "分析一下",
    "整理下",
    "整理一下",
    "说下",
    "说一下",
    "详细点",
    "具体点",
    "展开点",
    "展开说说",
}


def normalize_question_for_retrieval(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""

    q = q.replace("？", "").replace("?", "").replace("。", "").strip()

    for filler in sorted(QUERY_FILLERS, key=len, reverse=True):
        q = q.replace(filler, " ")

    q = re.sub(r"\s+", " ", q).strip()
    return q


def keep_only_allowed_terms(query: str, question: str, logger=None) -> str:
    """
    只保留“当前问题里本来就出现过”的词。
    rewrite 可以重排，但不允许新增词。
    """
    source_text = normalize_question_for_retrieval(question) or (question or "").strip()

    kept: list[str] = []
    dropped: list[str] = []
    seen: set[str] = set()

    for term in (query or "").split():
        t = term.strip()
        if not t or t in seen:
            continue

        if t in source_text:
            kept.append(t)
            seen.add(t)
        else:
            dropped.append(t)

    if dropped and logger:
        logger.info(f"🚫 [过滤新增词] {dropped}")

    return " ".join(kept)


def extract_strong_terms_from_question(question: str) -> list[str]:
    q = normalize_question_for_retrieval(question)
    if not q:
        return []

    result: list[str] = []

    def add(term: str):
        t = (term or "").strip()
        if t and t not in result:
            result.append(t)

    # 只提取问题里明确写出来的时间短语
    for pattern in [
        r"\d{4}年\d{1,2}月\d{1,2}[日号]?(?:后|之前|之后|以后)?",
        r"\d{1,2}月\d{1,2}[日号]?(?:后|之前|之后|以后)?",
        r"\d{1,2}[日号](?:后|之前|之后|以后)?",
        r"\d{1,2}:\d{2}",
    ]:
        for match in re.findall(pattern, q):
            add(match)

    # 只保留“当前问题中原样出现”的少量焦点词
    exact_terms = [
        "时间线",
        "经过",
        "过程",
        "详细点",
        "更详细",
        "法律性质",
        "性质",
        "合法吗",
        "是否合法",
        "合法",
        "合规吗",
        "合规",
        "动作",
        "做法",
        "行为",
        "处理",
        "公司",
        "对方",
        "之后",
        "后来",
        "后续",
    ]

    for term in exact_terms:
        if term in q:
            add(term)

    return result


def merge_rewritten_query_with_strong_terms(question: str, rewritten_query: str, logger=None) -> str:
    # rewrite 只能重排当前问题已有的词，不允许新增
    safe_rewritten = keep_only_allowed_terms(
        rewritten_query,
        question,
        logger=logger,
    )

    rewritten_terms = [x.strip() for x in safe_rewritten.split() if x.strip()]
    strong_terms = extract_strong_terms_from_question(question)

    merged: list[str] = []
    for term in rewritten_terms + strong_terms:
        if term and term not in merged:
            merged.append(term)

    result = " ".join(merged).strip()
    return result or (normalize_question_for_retrieval(question) or (question or "").strip())


def is_abstract_query(question: str) -> bool:
    q = normalize_question_for_retrieval(question)
    if not q:
        return True

    terms = extract_strong_terms_from_question(q)
    return not terms


def extract_timeline_evidence_from_chunks(relevant_indices, repo_state):
    date_patterns = [
        r"\d{4}年\d{1,2}月\d{1,2}日",
        r"\d{1,2}月\d{1,2}日",
        r"\d{1,2}日",
        r"\d{1,2}:\d{2}",
    ]

    results = []
    for idx in relevant_indices:
        text = repo_state.chunk_texts[idx]
        path = repo_state.chunk_paths[idx]

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in lines:
            if any(re.search(p, line) for p in date_patterns):
                results.append((path, line))

    seen = set()
    deduped = []
    for path, line in results:
        key = (path, line)
        if key not in seen:
            seen.add(key)
            deduped.append((path, line))

    return deduped


def build_timeline_evidence_text(timeline_items):
    if not timeline_items:
        return ""

    lines = [f"{path} | {line}" for path, line in timeline_items[:80]]
    return "\n".join(lines) + "\n\n"


def needs_timeline_evidence(question: str) -> bool:
    keywords = ["时间线", "经过", "过程", "梳理", "更详细", "详细点"]
    return any(x in question for x in keywords)


def is_result_expansion_followup(question: str) -> bool:
    q = normalize_question_for_retrieval(question)
    if not q:
        return False

    markers = {
        "更详细",
        "更详细的",
        "详细点",
        "具体点",
        "展开点",
        "展开说说",
        "继续",
        "然后呢",
        "后来呢",
        "扩大范围",
        "范围大点",
        "范围放宽",
        "放宽范围",
        "放宽一点",
        "扩大检索",
        "分析下",
        "分析一下",
        "法律性质",
        "性质",
        "合法吗",
        "是否合法",
    }

    return any(x in q for x in markers)


def is_related_record_listing_request(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    has_related = any(x in q for x in ["有关", "相关"])
    has_record_scope = any(x in q for x in ["记录", "文档", "文件"])
    has_listing = any(x in q for x in ["哪些", "哪几", "有哪", "最近"])
    return has_related and has_record_scope and has_listing


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


def _is_role_like_term(term_norm: str) -> bool:
    t = (term_norm or "").strip().lower()
    if not t:
        return False
    if t in ROLE_TERM_HINTS:
        return True
    if re.fullmatch(r"[a-z]{2,12}", t):
        return t in ROLE_TERM_HINTS or _looks_like_position_term(t)
    if _looks_like_position_term(t):
        return True
    if any(
        t.endswith(suffix)
        for suffix in ("负责人", "联系人", "经理", "主管", "总监", "专员", "顾问", "老师", "秘书", "主任", "委员", "代表")
    ):
        return True
    return False


def _expand_role_terms(role_terms: list[str]) -> list[str]:
    expanded: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        t = (token or "").strip()
        n = _normalize_lookup_token(t)
        if not n or n in seen:
            return
        seen.add(n)
        expanded.append(t)

    for token in role_terms or []:
        _add(token)

    return expanded


def _extract_company_from_line(line: str) -> str:
    raw = (line or "").strip()
    if not raw:
        return ""

    for pattern in COMPANY_LINE_PATTERNS:
        m = pattern.search(raw)
        if not m:
            continue
        company = _sanitize_company_candidate((m.group(1) or ""), raw_line=raw)
        if not company:
            continue
        return company
    return ""


def _sanitize_company_candidate(company_raw: str, *, raw_line: str = "") -> str:
    company = (company_raw or "").strip(" ·:：[]【】()（）")
    if len(company) < 2:
        return ""
    if any(bad in company for bad in COMPANY_BAN_TERMS):
        return ""
    if _looks_like_address_fragment(company):
        return ""
    if raw_line and re.search(r"(省|市|区|县).*\d+(?:层|楼|室|号|栋)", raw_line):
        return ""
    if re.search(r"[，,。；;：:、\s]{2,}", company):
        return ""
    if company.startswith(("对", "将", "能", "熟练", "负责", "推动", "提升")):
        return ""
    return company


def _extract_company_from_split_lines(current_line: str, next_line: str) -> str:
    cur = (current_line or "").strip()
    nxt = (next_line or "").strip()
    if not cur or not nxt:
        return ""
    if not re.search(r"[·•]$", cur):
        return ""
    if not re.search(
        r"^(?:hrbp|hr|recruiter|talent|招聘(?:经理|专员|顾问|官)?|猎头|人才顾问)\b",
        nxt,
        flags=re.IGNORECASE,
    ):
        return ""
    base = re.sub(r"[·•]+$", "", cur).strip()
    return _sanitize_company_candidate(base, raw_line=f"{cur} {nxt}")


def _looks_like_address_fragment(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    # 地址常见特征：行政区串 + 门牌楼层/房间号等
    if re.search(r"(省|市|区|县|路|街|号|层|室|栋|单元)", raw) and re.search(r"\d", raw):
        return True
    if re.search(r"(?:[^\s]{2,8}(?:省|市|区)){2,}", raw):
        return True
    if re.search(r"(省|市|区|县|路|街|大道|巷|弄)", raw) and re.search(r"(中心|大厦|园区|广场|楼)", raw):
        return True
    return False


def _extract_company_from_nearby_lines(lines: list[str], center: int, radius: int = 4) -> str:
    if not lines:
        return ""
    start = max(0, center - radius)
    end = min(len(lines), center + radius + 1)
    for i in range(start, end):
        company = _extract_company_from_line(lines[i])
        if company:
            return company
        if i + 1 < len(lines):
            company = _extract_company_from_split_lines(lines[i], lines[i + 1])
            if company:
                return company
    return ""


def _build_doc_text_lookup(repo_state) -> dict[str, str]:
    paths = list(getattr(repo_state, "paths", []) or [])
    docs = list(getattr(repo_state, "docs", []) or [])
    if not paths or not docs:
        return {}

    max_count = min(len(paths), len(docs))
    out: dict[str, str] = {}
    for i in range(max_count):
        path = str(paths[i] or "")
        text = str(docs[i] or "")
        if not path or not text:
            continue
        prev = out.get(path, "")
        if len(text) > len(prev):
            out[path] = text
    return out


def _normalized_bigrams(text: str) -> set[str]:
    src = (text or "").strip()
    if len(src) < 2:
        return set()
    return {src[i : i + 2] for i in range(len(src) - 1)}


def _role_line_match_score(raw_line: str, role_norms: list[str]) -> float:
    line_raw = (raw_line or "").strip()
    if not line_raw:
        return 0.0
    line_lower = line_raw.lower()
    line_norm = _normalize_lookup_token(line_raw)
    if not line_norm:
        return 0.0

    line_bigrams = _normalized_bigrams(line_norm)
    best = 0.0
    for role in role_norms:
        role_norm = _normalize_lookup_token(role)
        if not role_norm:
            continue

        if _term_matches_line(role_norm, line_lower, line_norm):
            exact_score = 1.0 + min(len(role_norm), 20) * 0.02
            if exact_score > best:
                best = exact_score
            continue

        if len(role_norm) < 4:
            continue
        role_bigrams = _normalized_bigrams(role_norm)
        if not role_bigrams or not line_bigrams:
            continue

        overlap_hits = len(role_bigrams & line_bigrams)
        if overlap_hits < 2:
            continue
        ratio = overlap_hits / max(len(role_bigrams), 1)
        if ratio < 0.45:
            continue
        fuzzy_score = 0.45 + ratio * 0.65
        if fuzzy_score > best:
            best = fuzzy_score

    return best


def _looks_like_company_hr_mapping_query(question: str, focus_terms: list[str]) -> tuple[bool, list[str]]:
    q = (question or "").strip()
    q_norm = _normalize_lookup_token(q)
    if not q_norm:
        return False, []

    role_terms: list[str] = []
    for token in focus_terms:
        norm = _normalize_lookup_token(token)
        if norm and _is_role_like_term(norm) and token not in role_terms:
            role_terms.append(token)
    if not role_terms:
        return False, []

    has_relation_marker = any(marker in q_norm for marker in DIRECT_LOOKUP_MARKERS)
    has_org_marker = any(marker in q for marker in COMPANY_QUERY_MARKERS)
    if not has_relation_marker:
        return False, []
    if not has_org_marker and not any(x in q for x in ("对应", "关联", "匹配", "分别")):
        return False, []

    return True, role_terms


def _looks_like_position_term(term_norm: str) -> bool:
    t = (term_norm or "").strip().lower()
    if not t:
        return False
    if re.fullmatch(r"[a-z]{4,32}", t):
        if len(t) >= 7 and re.search(r"(?:er|or)$", t):
            return True
        if re.search(r"(?:ist|ian|ant|ive|ary|eer|ician|ologist|ographer)$", t):
            return True
    return any(
        t.endswith(suffix)
        for suffix in (
            "工程师",
            "架构师",
            "分析师",
            "研究员",
            "开发",
            "测试",
            "算法",
            "经理",
            "主管",
            "总监",
            "顾问",
            "职位",
            "岗位",
        )
    )


def _looks_like_mapping_followup_query(
    question: str,
    focus_terms: list[str],
    *,
    allow_followup_inference: bool,
) -> bool:
    if not allow_followup_inference:
        return False

    q = (question or "").strip()
    q_norm = _normalize_lookup_token(q)
    if not q_norm:
        return False

    if any(term in q_norm for term in MAPPING_FOLLOWUP_BLOCK_TERMS):
        return False

    if any(marker in q for marker in ROLE_NAME_QUERY_MARKERS):
        return False

    has_followup_signal = any(marker in q_norm for marker in DIRECT_LOOKUP_FOLLOWUP_MARKERS)
    has_selector_signal = bool(DIRECT_LOOKUP_SELECTOR_PATTERN.search(q)) or bool(RANGE_SIGNATURE_PATTERN.search(q))
    has_relation_signal = (
        any(marker in q for marker in COMPANY_QUERY_MARKERS)
        or any(marker in q_norm for marker in ("对应", "关联", "匹配", "映射"))
        or ("hr" in q_norm)
    )

    focus_norms = {_normalize_lookup_token(t) for t in focus_terms if _normalize_lookup_token(t)}
    has_anchor_focus = any(
        norm
        and norm not in DIRECT_LOOKUP_GENERIC_ANCHOR_TERMS
        and not _is_role_like_term(norm)
        for norm in focus_norms
    )
    has_position_focus = any(_looks_like_position_term(norm) for norm in focus_norms)

    if not (has_followup_signal or has_selector_signal):
        return False
    if not has_relation_signal:
        return False
    return has_anchor_focus or has_position_focus


def _looks_like_role_name_query(question: str, focus_terms: list[str]) -> tuple[bool, list[str]]:
    q = (question or "").strip()
    q_norm = _normalize_lookup_token(q)
    if not q_norm:
        return False, []

    role_terms: list[str] = []
    for token in focus_terms:
        norm = _normalize_lookup_token(token)
        if norm and _is_role_like_term(norm) and token not in role_terms:
            role_terms.append(token)
    if not role_terms:
        return False, []

    if any(marker in q for marker in ROLE_NON_NAME_QUERY_MARKERS):
        return False, []

    focus_norms = {_normalize_lookup_token(t) for t in focus_terms if _normalize_lookup_token(t)}
    role_norms = [_normalize_lookup_token(t) for t in role_terms if _normalize_lookup_token(t)]
    target_role_norms = [n for n in role_norms if not _looks_like_position_term(n)]
    has_anchor_focus = any(
        norm
        and norm not in DIRECT_LOOKUP_GENERIC_ANCHOR_TERMS
        and (not _is_role_like_term(norm) or _looks_like_position_term(norm))
        for norm in focus_norms
    )

    has_name_marker = any(marker in q for marker in ROLE_NAME_QUERY_MARKERS)
    has_short_followup_marker = len(q_norm) <= 10 and any(marker in q for marker in ROLE_NAME_FOLLOWUP_MARKERS)
    has_possessive_role_marker = (
        ("的" in q)
        and has_anchor_focus
        and bool(target_role_norms)
        and any(role_norm in q_norm for role_norm in target_role_norms)
    )
    if not has_name_marker and not has_short_followup_marker and not has_possessive_role_marker:
        return False, []

    return True, role_terms


def _extract_role_name_items(
    *,
    role_terms: list[str],
    anchor_terms: list[str],
    required_selector_signatures: set[str] | None = None,
    relevant_indices,
    repo_state,
    max_items: int,
) -> list[dict]:
    role_norms = [_normalize_lookup_token(t) for t in role_terms if _normalize_lookup_token(t)]
    if not role_norms:
        return []

    anchor_norms = [_normalize_lookup_token(t) for t in anchor_terms if _normalize_lookup_token(t)]
    candidates: list[dict] = []
    seen: set[tuple[str, str]] = set()
    processed_paths: set[str] = set()
    doc_text_lookup = _build_doc_text_lookup(repo_state)

    for idx in relevant_indices or []:
        try:
            path = str(repo_state.chunk_paths[idx])
            if path in processed_paths:
                continue
            processed_paths.add(path)
            chunk_text = repo_state.chunk_texts[idx] or ""
            text = doc_text_lookup.get(path) or chunk_text
        except Exception:
            continue

        lines = [ln.strip() for ln in text.splitlines() if (ln or "").strip()]
        if not lines:
            continue
        text_lower = "\n".join(lines).lower()
        text_norm = _normalize_lookup_token("\n".join(lines))
        if required_selector_signatures:
            file_selector_sigs = _extract_selector_signatures(text_lower)
            if not _selector_constraints_satisfied(file_selector_sigs, set(required_selector_signatures)):
                continue
        file_anchor_hits = 0
        if anchor_norms:
            for term in anchor_norms:
                if _term_matches_text(term, text_lower, text_norm):
                    file_anchor_hits += 1
            min_file_anchor_hits = 2 if len(anchor_norms) >= 3 else 1
            if file_anchor_hits < min_file_anchor_hits:
                continue

        for i, line in enumerate(lines):
            line_norm = _normalize_lookup_token(line)
            if not line_norm:
                continue
            if not any(_term_matches_line(t, line.lower(), line_norm) for t in role_norms):
                continue

            for wi in range(i - 4, i + 5):
                if wi < 0 or wi >= len(lines):
                    continue
                evidence = lines[wi]
                if not evidence or len(evidence) > 180:
                    continue

                for pattern in PERSON_NAME_PATTERNS:
                    for m in pattern.findall(evidence):
                        name = (m or "").strip()
                        if not name:
                            continue
                        if not _is_plausible_person_name(name, evidence):
                            continue
                        key = (path, name)
                        if key in seen:
                            continue
                        seen.add(key)

                        score = 1.0
                        if wi == i:
                            score += 0.6
                        if "·" in evidence or ":" in evidence or "：" in evidence:
                            score += 0.2
                        if re.search(r"(先生|女士|老师)$", name):
                            score += 0.4
                        if file_anchor_hits:
                            score += min(file_anchor_hits, 4) * 0.3
                        elif anchor_norms:
                            score *= 0.4
                        company = _extract_company_from_nearby_lines(lines, i, radius=5)
                        if company:
                            score += 0.25

                        candidates.append(
                            {
                                "name": name,
                                "path": path,
                                "evidence": evidence,
                                "company": company,
                                "score": score,
                            }
                        )

    candidates.sort(key=lambda x: (-float(x["score"]), x["path"], x["name"]))
    if anchor_norms and candidates:
        best_score = float(candidates[0]["score"])
        candidates = [x for x in candidates if float(x["score"]) >= best_score - 0.25]
    return candidates[:max_items]


def _extract_company_hr_mapping_items(
    *,
    role_terms: list[str],
    anchor_terms: list[str],
    required_selector_signatures: set[str],
    relevant_indices,
    repo_state,
    max_items: int,
) -> list[dict]:
    role_norms = [_normalize_lookup_token(t) for t in role_terms if _normalize_lookup_token(t)]
    specific_role_norms = [t for t in role_norms if t not in GENERIC_ROLE_NORMS and len(t) >= 2]
    anchor_norms = [_normalize_lookup_token(t) for t in anchor_terms if _normalize_lookup_token(t)]
    require_role_match = bool(role_norms)

    items: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    processed_paths: set[str] = set()
    doc_text_lookup = _build_doc_text_lookup(repo_state)

    for idx in relevant_indices or []:
        try:
            path = str(repo_state.chunk_paths[idx])
            if path in processed_paths:
                continue
            processed_paths.add(path)
            chunk_text = repo_state.chunk_texts[idx] or ""
            text = doc_text_lookup.get(path) or chunk_text
        except Exception:
            continue

        lines = [ln.strip() for ln in text.splitlines() if (ln or "").strip()]
        if not lines:
            continue

        file_text_lower = "\n".join(lines).lower()
        file_text_norm = _normalize_lookup_token("\n".join(lines))
        file_anchor_hits = 0
        for term in anchor_norms:
            if _term_matches_text(term, file_text_lower, file_text_norm):
                file_anchor_hits += 1
        min_file_anchor_hits = 2 if len(anchor_norms) >= 3 else 1
        if anchor_norms and file_anchor_hits < min_file_anchor_hits:
            continue
        if required_selector_signatures:
            file_selector_sigs = _extract_selector_signatures(file_text_lower)
            if not _selector_constraints_satisfied(file_selector_sigs, required_selector_signatures):
                continue

        company_candidates: list[tuple[int, str]] = []
        person_candidates: list[tuple[int, str, str]] = []
        heading_candidates: list[tuple[int, str]] = []
        role_match_scores: dict[int, float] = {}
        specific_role_scores: dict[int, float] = {}
        for li, li_text in enumerate(lines):
            company = _extract_company_from_line(li_text)
            if not company and li + 1 < len(lines):
                company = _extract_company_from_split_lines(li_text, lines[li + 1])
            if company:
                company_candidates.append((li, company))
            if _looks_like_heading_line(li_text):
                heading_candidates.append((li, li_text))
            if require_role_match:
                role_score = _role_line_match_score(li_text, role_norms)
                target_role_score = (
                    _role_line_match_score(li_text, specific_role_norms)
                    if specific_role_norms
                    else role_score
                )
                if target_role_score >= 0.55:
                    role_match_scores[li] = role_score
                    specific_role_scores[li] = target_role_score
            for pattern in PERSON_NAME_PATTERNS:
                for m in pattern.findall(li_text):
                    name = (m or "").strip()
                    if name and _is_plausible_person_name(name, li_text):
                        person_candidates.append((li, name, li_text))

        if require_role_match and not role_match_scores:
            continue
        if not company_candidates or not person_candidates:
            continue

        file_item_count_before = len(items)
        role_score_terms = specific_role_norms or role_norms

        def _nearest_company(center: int, max_dist: int) -> tuple[str, int]:
            best_name = ""
            best_dist = 10_000
            for ci, cname in company_candidates:
                dist = abs(ci - center)
                if dist <= max_dist and dist < best_dist:
                    best_dist = dist
                    best_name = cname
            return best_name, best_dist

        def _nearest_person(center: int, max_dist: int) -> tuple[str, str, int]:
            best_name = ""
            best_line = ""
            best_dist = 10_000
            for pi, pname, pline in person_candidates:
                dist = abs(pi - center)
                if dist <= max_dist and dist < best_dist:
                    best_dist = dist
                    best_name = pname
                    best_line = pline
            return best_name, best_line, best_dist

        def _best_heading(center: int) -> str:
            best_line = ""
            best_dist = 10_000
            for hi, hline in heading_candidates:
                dist = abs(hi - center)
                if dist <= 12 and dist < best_dist:
                    best_dist = dist
                    best_line = hline
            if best_line:
                return best_line
            return lines[center]

        def _best_role_line(center: int) -> tuple[str, float]:
            best_line = _best_heading(center)
            best_rank = _role_line_match_score(best_line, role_score_terms) if role_score_terms else 0.0
            start = max(0, center - 10)
            end = min(len(lines), center + 11)
            for li in range(start, end):
                li_text = lines[li]
                role_score = _role_line_match_score(li_text, role_score_terms) if role_score_terms else 0.0
                if require_role_match and role_score <= 0.0:
                    continue
                rank = role_score
                if _looks_like_heading_line(li_text):
                    rank += 0.22
                if _looks_like_detail_line(li_text):
                    rank -= 0.18
                rank -= max(0, abs(li - center) - 1) * 0.08
                if rank > best_rank:
                    best_rank = rank
                    best_line = li_text
            return best_line, max(0.0, _role_line_match_score(best_line, role_score_terms) if role_score_terms else 0.0)

        if require_role_match:
            candidate_centers = sorted(role_match_scores.keys(), key=lambda i: (-role_match_scores[i], i))
        elif anchor_norms:
            candidate_centers = [
                i
                for i, line in enumerate(lines)
                if _normalize_lookup_token(line)
                and any(_term_matches_line(t, line.lower(), _normalize_lookup_token(line)) for t in anchor_norms)
            ]
        else:
            candidate_centers = list(range(len(lines)))

        for i in candidate_centers:
            line = lines[i]
            line_norm = _normalize_lookup_token(line)
            if not line_norm:
                continue

            role_match_score = role_match_scores.get(i, 0.0)
            if require_role_match and role_match_score <= 0.0:
                continue
            role_specific_score = specific_role_scores.get(i, role_match_score)
            if require_role_match and specific_role_norms and role_specific_score <= 0.0:
                continue

            line_has_anchor = any(_term_matches_line(t, line.lower(), line_norm) for t in anchor_norms) if anchor_norms else False
            if not require_role_match and anchor_norms and not line_has_anchor:
                continue

            local_lines = lines[max(0, i - 20): min(len(lines), i + 21)]
            local_lower = "\n".join(local_lines).lower()
            local_norm = _normalize_lookup_token("\n".join(local_lines))
            local_anchor_hits = 0
            for term in anchor_norms:
                if _term_matches_text(term, local_lower, local_norm):
                    local_anchor_hits += 1

            company, company_dist = _nearest_company(i, 8)
            used_company_fallback = False
            if not company:
                company, company_dist = _nearest_company(i, 60)
                used_company_fallback = bool(company)
            if not company:
                continue

            nearest_name, nearest_name_line, nearest_dist = _nearest_person(i, 12)
            used_name_fallback = False
            if not nearest_name:
                nearest_name, nearest_name_line, nearest_dist = _nearest_person(i, 80)
                used_name_fallback = bool(nearest_name)
            if not nearest_name:
                continue

            role_line, role_line_match_score = _best_role_line(i)
            if anchor_norms:
                role_line_norm = _normalize_lookup_token(role_line)
                if not any(_term_matches_line(t, role_line.lower(), role_line_norm) for t in anchor_norms):
                    best_anchor_line = ""
                    best_anchor_rank = -1.0
                    for li in local_lines:
                        li_norm = _normalize_lookup_token(li)
                        if not li_norm:
                            continue
                        if not any(_term_matches_line(t, li.lower(), li_norm) for t in anchor_norms):
                            continue
                        li_role_score = _role_line_match_score(li, role_score_terms) if role_score_terms else 0.0
                        if require_role_match and li_role_score < 0.2:
                            continue
                        li_rank = li_role_score + (0.2 if _looks_like_heading_line(li) else 0.0)
                        if li_rank > best_anchor_rank:
                            best_anchor_rank = li_rank
                            best_anchor_line = li
                    if best_anchor_line:
                        role_line = best_anchor_line
                        role_line_match_score = _role_line_match_score(role_line, role_score_terms) if role_score_terms else 0.0

            score = 1.0
            score += min(file_anchor_hits, 6) * 0.24
            score += min(local_anchor_hits, 5) * 0.40
            score += max(0, 1.1 - nearest_dist * 0.14)
            score += max(0, 0.7 - company_dist * 0.09)
            score += min(role_specific_score, 1.8) * 0.75
            score += min(role_line_match_score, 1.6) * 0.55
            if used_company_fallback:
                score -= 0.20
            if used_name_fallback:
                score -= 0.20
            if role_line and len(role_line) <= 100:
                score += 0.15
            if _looks_like_detail_line(role_line):
                score -= 0.25
            if require_role_match and role_line_match_score < 0.45:
                score -= 0.45

            key = (path, company, nearest_name)
            if key in seen:
                continue
            seen.add(key)

            items.append(
                {
                    "path": path,
                    "company": company,
                    "name": nearest_name,
                    "role_line": role_line,
                    "evidence": nearest_name_line,
                    "score": score,
                }
            )

        if (not require_role_match) and len(items) == file_item_count_before:
            # Fallback: allow file-level pairing when OCR layout breaks local adjacency.
            pair = None
            best_dist = 10_000
            for ci, cname in company_candidates:
                for pi, pname, pline in person_candidates:
                    dist = abs(ci - pi)
                    if dist < best_dist:
                        best_dist = dist
                        pair = (cname, pname, pline, pi)
            if pair is not None:
                company, pname, pline, pidx = pair
                role_line = _best_heading(pidx)
                local_lines = lines[max(0, pidx - 20): min(len(lines), pidx + 21)]
                local_lower = "\n".join(local_lines).lower()
                local_norm = _normalize_lookup_token("\n".join(local_lines))
                local_anchor_hits = 0
                for term in anchor_norms:
                    if _term_matches_text(term, local_lower, local_norm):
                        local_anchor_hits += 1
                key = (path, company, pname)
                if key not in seen:
                    seen.add(key)
                    score = 0.9 + min(file_anchor_hits, 6) * 0.22 + min(local_anchor_hits, 4) * 0.3
                    score += max(0, 0.6 - best_dist * 0.06)
                    items.append(
                        {
                            "path": path,
                            "company": company,
                            "name": pname,
                            "role_line": role_line,
                            "evidence": pline,
                            "score": score,
                        }
                    )

    items.sort(key=lambda x: (-float(x["score"]), x["path"], x["company"], x["name"]))
    selected: list[dict] = []
    used_paths: set[str] = set()
    cap = min(max_items, 4)
    for item in items:
        path = str(item.get("path") or "")
        if not path:
            continue
        if path in used_paths:
            continue
        used_paths.add(path)
        selected.append(item)
        if len(selected) >= cap:
            break
    return selected


def _looks_like_direct_lookup_question(question: str) -> bool:
    q = _normalize_lookup_token(question)
    if not q:
        return False
    if any(marker in q for marker in DIRECT_LOOKUP_NON_LOOKUP_MARKERS):
        return False
    if any(marker in q for marker in DIRECT_LOOKUP_ANALYSIS_MARKERS):
        return False
    has_lookup_signal = any(marker in q for marker in DIRECT_LOOKUP_MARKERS)
    has_focus_signal = bool(re.search(r"[a-zA-Z0-9\u4e00-\u9fa5]{2,}", question or ""))
    return has_lookup_signal and has_focus_signal


def _looks_like_direct_lookup_followup_question(question: str) -> bool:
    q = _normalize_lookup_token(question)
    if not q:
        return False
    if _looks_like_direct_lookup_question(question):
        return True
    if any(marker in q for marker in DIRECT_LOOKUP_NON_LOOKUP_MARKERS):
        return False
    if any(marker in q for marker in DIRECT_LOOKUP_ANALYSIS_MARKERS):
        return False
    has_followup_signal = any(marker in q for marker in DIRECT_LOOKUP_FOLLOWUP_MARKERS)
    has_focus_signal = bool(re.search(r"[a-zA-Z0-9\u4e00-\u9fa5]{2,}", question or ""))
    has_selector_signal = bool(DIRECT_LOOKUP_SELECTOR_PATTERN.search(question or ""))
    return has_followup_signal and (has_focus_signal or has_selector_signal)


def _extract_direct_lookup_terms(question: str, search_query: str) -> list[str]:
    from retrieval.query_utils import extract_query_terms

    terms: list[str] = []
    seen: set[str] = set()

    for raw in extract_query_terms(search_query or "", question or ""):
        token = (raw or "").strip()
        norm = _normalize_lookup_token(token)
        if not norm:
            continue
        if norm in DIRECT_LOOKUP_STOP_TERMS:
            continue
        if len(norm) <= 1:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        terms.append(token)

    if terms:
        return terms

    raw_terms = re.findall(r"[a-zA-Z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", question or "")
    for token in raw_terms:
        norm = _normalize_lookup_token(token)
        if not norm or norm in DIRECT_LOOKUP_STOP_TERMS or norm in seen:
            continue
        seen.add(norm)
        terms.append(token)
    return terms


def _extract_direct_lookup_focus_terms(question: str) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    for token in re.findall(r"[a-zA-Z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", question or ""):
        norm = _normalize_lookup_token(token)
        if not norm:
            continue
        if norm in DIRECT_LOOKUP_STOP_TERMS:
            continue
        if len(norm) <= 1:
            continue
        if norm in seen:
            continue
        seen.add(norm)
        terms.append(token)
    return terms


def _build_direct_lookup_evidence_items(
    *,
    terms: list[str],
    focus_terms: list[str],
    relevant_indices,
    repo_state,
    max_items: int,
) -> list[dict]:
    term_norms = [_normalize_lookup_token(t) for t in terms if _normalize_lookup_token(t)]
    if not term_norms:
        return []
    selector_query_hit = any(
        re.fullmatch(r"\d{1,4}", t) or bool(RANGE_SIGNATURE_PATTERN.search(t))
        for t in term_norms
    )

    focus_norms = {_normalize_lookup_token(t) for t in focus_terms if _normalize_lookup_token(t)}
    has_position_focus = any(_looks_like_position_term(norm) for norm in focus_norms)
    weighted_terms: dict[str, float] = {}
    for t in term_norms:
        if t in weighted_terms:
            continue
        weighted_terms[t] = 2.0 if t in focus_norms else 1.0

    ranked: list[dict] = []
    seen_lines: set[tuple[str, str]] = set()

    for idx in relevant_indices or []:
        try:
            path = repo_state.chunk_paths[idx]
            text = repo_state.chunk_texts[idx] or ""
        except Exception:
            continue

        per_path_items: list[dict] = []
        for line in text.splitlines():
            raw_line = (line or "").strip()
            if not raw_line:
                continue
            if len(raw_line) > 160:
                continue

            raw_line_lower = raw_line.lower()
            line_norm = _normalize_lookup_token(raw_line)
            if not line_norm:
                continue

            hits = 0
            score = 0.0
            matched_focus = False
            for t in term_norms:
                if _term_matches_line(t, raw_line_lower, line_norm):
                    hits += 1
                    weight = weighted_terms.get(t, 1.0)
                    score += (0.8 + min(len(t), 10) * 0.08) * weight
                    if t in focus_norms:
                        matched_focus = True
            if hits <= 0:
                continue

            if hits >= 2:
                score += 0.35
            if DIRECT_LOOKUP_STRUCTURED_LINE_PATTERN.search(raw_line):
                score += 0.18
            if 6 <= len(raw_line) <= 80:
                score += 0.12
            if RANGE_SIGNATURE_PATTERN.search(raw_line):
                score += 0.42
                if selector_query_hit:
                    score += 0.32
            if _looks_like_heading_line(raw_line):
                score += 0.28
            if _looks_like_detail_line(raw_line):
                score -= 0.18
            if has_position_focus:
                line_tokens = re.findall(r"[a-zA-Z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", raw_line)
                line_has_position = any(
                    _looks_like_position_term(_normalize_lookup_token(tok))
                    for tok in line_tokens
                )
                matched_position_term = any(
                    _looks_like_position_term(t) and _term_matches_line(t, raw_line_lower, line_norm)
                    for t in term_norms
                )
                if not line_has_position and not matched_position_term:
                    continue
                if not line_has_position and not _looks_like_heading_line(raw_line):
                    score -= 0.25

            if focus_norms and matched_focus:
                score += 0.45
            elif focus_norms and not matched_focus:
                score *= 0.62

            per_path_items.append(
                {
                    "path": str(path),
                    "line": raw_line,
                    "line_norm": line_norm,
                    "score": score,
                    "matched_focus": matched_focus,
                }
            )

        if not per_path_items:
            continue

        per_path_items.sort(key=lambda x: (-float(x["score"]), x["line"]))
        for item in per_path_items[:3]:
            dedup_key = (item["path"], item["line_norm"])
            if dedup_key in seen_lines:
                continue
            seen_lines.add(dedup_key)
            ranked.append(
                {
                    "path": item["path"],
                    "line": item["line"],
                    "score": item["score"],
                    "matched_focus": item["matched_focus"],
                }
            )

    ranked.sort(key=lambda x: (-float(x["score"]), x["path"], x["line"]))

    def _select_diverse(items: list[dict]) -> list[dict]:
        selected: list[dict] = []
        used_paths: set[str] = set()

        for item in items:
            path = str(item.get("path") or "")
            if not path or path in used_paths:
                continue
            used_paths.add(path)
            selected.append(item)
            if len(selected) >= max_items:
                return selected

        return selected

    focus_ranked = [x for x in ranked if x.get("matched_focus")]
    if focus_norms and focus_ranked:
        return _select_diverse(focus_ranked)
    return _select_diverse(ranked)


def maybe_build_direct_lookup_answer(
    *,
    question: str,
    search_query: str,
    relevant_indices,
    repo_state,
    max_items: int = 6,
    logger=None,
    allow_followup_inference: bool = False,
    force_local_evidence: bool = False,
) -> str | None:
    focus_terms = _extract_direct_lookup_focus_terms(question)
    is_role_name_query, _ = _looks_like_role_name_query(question, focus_terms)
    if (not force_local_evidence) and (not _looks_like_direct_lookup_question(question)):
        if not is_role_name_query:
            if not allow_followup_inference:
                return None
            if not _looks_like_direct_lookup_followup_question(question):
                return None

    terms = _extract_direct_lookup_terms(question, search_query)
    if not terms:
        return None
    anchor_terms: list[str] = []
    for token in terms:
        norm = _normalize_lookup_token(token)
        if not norm:
            continue
        if _is_role_like_term(norm):
            continue
        if norm in {"公司", "企业", "单位"}:
            continue
        if norm in DIRECT_LOOKUP_GENERIC_ANCHOR_TERMS:
            continue
        if token not in anchor_terms:
            anchor_terms.append(token)

    is_company_hr_mapping_query, role_terms_for_mapping = _looks_like_company_hr_mapping_query(question, focus_terms)
    if is_company_hr_mapping_query:
        role_terms_for_mapping = _expand_role_terms(role_terms_for_mapping)
        required_selector_signatures = _extract_selector_signatures(question)
        if not required_selector_signatures:
            required_selector_signatures = _extract_selector_signatures(search_query)
        mapping_items = _extract_company_hr_mapping_items(
            role_terms=role_terms_for_mapping,
            anchor_terms=anchor_terms,
            required_selector_signatures=required_selector_signatures,
            relevant_indices=relevant_indices,
            repo_state=repo_state,
            max_items=max_items,
        )
        if mapping_items:
            lines = ["根据当前检索片段，匹配到以下“公司-HR”对应关系："]
            for i, item in enumerate(mapping_items, start=1):
                lines.append(f"{i}. {item['company']}（HR：{item['name']}）")
                if item["role_line"]:
                    lines.append(f"   关联线索：{item['role_line']}")
                lines.append(f"   来源：{item['path']}")
            if logger:
                logger.info(f"🧪 [直接检索稳态回答] 公司-HR对应命中 {len(mapping_items)} 条")
            return "\n".join(lines)

    if _looks_like_mapping_followup_query(
        question,
        focus_terms,
        allow_followup_inference=allow_followup_inference,
    ):
        inferred_role_terms = _expand_role_terms(
            [token for token in focus_terms if _is_role_like_term(_normalize_lookup_token(token))]
        )
        required_selector_signatures = _extract_selector_signatures(question)
        if not required_selector_signatures:
            required_selector_signatures = _extract_selector_signatures(search_query)
        mapping_items = _extract_company_hr_mapping_items(
            role_terms=inferred_role_terms,
            anchor_terms=anchor_terms,
            required_selector_signatures=required_selector_signatures,
            relevant_indices=relevant_indices,
            repo_state=repo_state,
            max_items=max_items,
        )
        if mapping_items:
            lines = ["根据当前检索片段，按上一轮口径匹配到以下“公司-HR”对应关系："]
            for i, item in enumerate(mapping_items, start=1):
                lines.append(f"{i}. {item['company']}（HR：{item['name']}）")
                if item["role_line"]:
                    lines.append(f"   关联线索：{item['role_line']}")
                lines.append(f"   来源：{item['path']}")
            if logger:
                logger.info(f"🧪 [直接检索稳态回答] 追问口径继承-公司HR命中 {len(mapping_items)} 条")
            return "\n".join(lines)

    is_role_name_query, role_terms = _looks_like_role_name_query(question, focus_terms)
    if is_role_name_query:
        role_terms = _expand_role_terms(role_terms)
        required_selector_signatures = _extract_selector_signatures(question)
        if not required_selector_signatures:
            required_selector_signatures = _extract_selector_signatures(search_query)
        q_norm = _normalize_lookup_token(question)
        role_max_items = max_items
        is_singular_owner_query = (
            ("的" in (question or ""))
            and not any(marker in (question or "") for marker in ("哪些", "哪几", "分别", "列表", "清单", "和", "以及", "及", "/", "、"))
        )
        if is_singular_owner_query:
            role_max_items = 1
        if allow_followup_inference and len(q_norm) <= 8:
            role_max_items = min(max_items, 4)
        role_name_items = _extract_role_name_items(
            role_terms=role_terms,
            anchor_terms=anchor_terms,
            required_selector_signatures=required_selector_signatures,
            relevant_indices=relevant_indices,
            repo_state=repo_state,
            max_items=role_max_items,
        )
        if not role_name_items and anchor_terms:
            role_name_items = _extract_role_name_items(
                role_terms=role_terms,
                anchor_terms=[],
                required_selector_signatures=required_selector_signatures,
                relevant_indices=relevant_indices,
                repo_state=repo_state,
                max_items=role_max_items,
            )
        if role_name_items:
            role_label = " / ".join(role_terms[:2])
            lines = [f"根据当前检索片段，匹配到 {role_label} 相关姓名："]
            for i, item in enumerate(role_name_items, start=1):
                if item.get("company"):
                    lines.append(f"{i}. {item['name']}（{item['company']}）")
                else:
                    lines.append(f"{i}. {item['name']}")
                lines.append(f"   来源：{item['path']}")
                lines.append(f"   证据：{item['evidence']}")
            if logger:
                logger.info(f"🧪 [直接检索稳态回答] 角色姓名命中 {len(role_name_items)} 条")
            return "\n".join(lines)

    items = _build_direct_lookup_evidence_items(
        terms=terms,
        focus_terms=focus_terms,
        relevant_indices=relevant_indices,
        repo_state=repo_state,
        max_items=max_items,
    )
    if not items:
        if force_local_evidence:
            return "根据当前检索片段，暂未提取到稳定的可核对条目。可继续说“看下1/看下2”查看来源文件。"
        if allow_followup_inference and focus_terms:
            focus_tip = "、".join(focus_terms[:3])
            return f"当前检索片段未直接命中“{focus_tip}”相关证据，先不给出推断；可补充更完整关键词后再查。"
        return None

    lines = ["根据当前检索片段，先给你可直接核对的证据："]
    for i, item in enumerate(items, start=1):
        lines.append(f"{i}. {item['line']}")
        lines.append(f"   来源：{item['path']}")

    if logger:
        logger.info(f"🧪 [直接检索稳态回答] 命中 {len(items)} 条证据")

    return "\n".join(lines)


FILE_LOOKUP_POLITE_PREFIXES = (
    "帮我查下",
    "帮我查找",
    "帮我查询",
    "帮我找下",
    "帮我看看",
    "帮我看下",
    "帮我定位下",
    "帮我定位一下",
    "帮我",
    "麻烦你",
    "麻烦",
    "请你",
    "请",
    "查下",
    "查找",
    "查询",
    "看看",
    "看下",
    "定位下",
    "定位一下",
)

FILE_LOOKUP_GENERIC_TERMS = {
    "文件",
    "文档",
    "记录",
    "哪个",
    "哪些",
    "哪份",
    "哪一份",
    "在哪",
    "在哪个",
    "是在",
    "是在哪",
    "文件名",
    "文档名",
    "记录名",
    "帮我",
    "查下",
    "查找",
    "查询",
    "看看",
    "看下",
}

FILE_LOOKUP_FOLLOWUP_PRONOUNS = {
    "这",
    "那",
    "这个",
    "那个",
    "它",
    "他",
    "她",
    "它们",
    "他们",
    "这条",
    "那条",
    "这个岗位",
    "那个岗位",
    "这个职位",
    "那个职位",
    "这个公司",
    "那个公司",
}


def _normalize_lookup_token(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fa5]+", "", (text or "").lower())


def _strip_file_lookup_prefix(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""

    changed = True
    while changed and q:
        changed = False
        for prefix in FILE_LOOKUP_POLITE_PREFIXES:
            if q.startswith(prefix):
                q = q[len(prefix):].strip(" ，,：:")
                changed = True
                break
    return q


def _extract_file_lookup_target(question: str) -> str:
    q = _strip_file_lookup_prefix(question)
    if not q:
        return ""

    compact = re.sub(r"\s+", "", q)
    m = re.search(
        r"(?P<target>.+?)(?:是)?(?:在)?(?:哪(?:个|些|份)?(?:文件|文档|记录)|哪个(?:文件|文档|记录))",
        compact,
    )
    if not m:
        return ""

    target = (m.group("target") or "").strip(" ，,：:。！？!?")
    if not target:
        return ""

    for prefix in FILE_LOOKUP_POLITE_PREFIXES:
        if target.startswith(prefix):
            target = target[len(prefix):].strip(" ，,：:")
    return target


def _extract_followup_file_lookup_target(question: str) -> str:
    q = _strip_file_lookup_prefix(question)
    if not q:
        return ""

    q = re.sub(r"\s+", "", q)
    q = q.strip("，,：:。！？!?`\"'[]【】")
    q = re.sub(r"(?:呢|吗|嘛|呀|啊)+$", "", q)
    q = re.sub(r"(?:怎么样|如何|咋样|怎么说|怎么理解)$", "", q)
    q = q.strip("，,：:。！？!?`\"'[]【】")
    if not q:
        return ""

    q_norm = _normalize_lookup_token(q)
    if not q_norm:
        return ""
    if q_norm in FILE_LOOKUP_GENERIC_TERMS:
        return ""
    if q_norm in FILE_LOOKUP_FOLLOWUP_PRONOUNS:
        return ""
    if re.fullmatch(r"\d+(?:\.\d+)?", q_norm):
        return ""
    if len(q_norm) <= 1:
        return ""
    if not re.search(r"[a-zA-Z\u4e00-\u9fa5]", q):
        return ""
    return q


def _build_file_lookup_terms(question: str, search_query: str, target: str) -> list[str]:
    # Local import to keep this module independent from retrieval bootstrap order.
    from retrieval.query_utils import extract_query_terms

    raw_terms = list(extract_query_terms(search_query or "", question or ""))
    if target:
        raw_terms.insert(0, target)
        target_norm = _normalize_lookup_token(target)
        if target_norm and target_norm != target:
            raw_terms.insert(1, target_norm)

        for token in re.findall(r"[a-z0-9_]{2,}|[\u4e00-\u9fa5]{2,}", target.lower()):
            raw_terms.append(token)

    terms: list[str] = []
    seen: set[str] = set()
    for raw in raw_terms:
        t = (raw or "").strip()
        if not t:
            continue
        t_norm = _normalize_lookup_token(t)
        if not t_norm:
            continue
        if t_norm in FILE_LOOKUP_GENERIC_TERMS:
            continue
        if t_norm.startswith(("帮我", "请", "麻烦")):
            continue
        if ("文件" in t_norm or "文档" in t_norm or "记录" in t_norm) and ("哪" in t_norm or "在" in t_norm):
            continue
        if len(t_norm) <= 1:
            continue
        if t_norm in seen:
            continue
        seen.add(t_norm)
        terms.append(t)
    return terms


def maybe_build_file_location_answer(
    *,
    question: str,
    search_query: str,
    relevant_indices,
    repo_state,
    max_items: int = 5,
    logger=None,
    allow_followup_inference: bool = False,
) -> str | None:
    # Local import to avoid any potential cross-module initialization coupling.
    from retrieval.search_intent import is_file_location_lookup_query

    if not relevant_indices:
        return None

    is_direct_lookup = is_file_location_lookup_query(question, search_query)
    target = _extract_file_lookup_target(question)
    if not is_direct_lookup:
        if not allow_followup_inference:
            return None
        if not target:
            target = _extract_followup_file_lookup_target(question)
        if not target:
            return None
        if logger:
            logger.info(f"   🧷 [文件定位追问推断] target={target}")

    terms = _build_file_lookup_terms(question, search_query, target)
    if not terms and not target:
        return None

    file_records: dict[str, dict] = {}
    for idx in relevant_indices:
        try:
            path = repo_state.chunk_paths[idx]
            text = repo_state.chunk_texts[idx] or ""
        except Exception:
            continue
        rec = file_records.setdefault(path, {"texts": []})
        if text and len(rec["texts"]) < 4:
            rec["texts"].append(text)

    if not file_records:
        return None

    norm_by_file: dict[str, tuple[str, str]] = {}
    df_by_term: dict[str, int] = {}
    term_norm_cache: dict[str, str] = {}

    for path, rec in file_records.items():
        merged_text = "\n".join(rec["texts"])
        text_norm = _normalize_lookup_token(merged_text)
        path_norm = _normalize_lookup_token(path)
        norm_by_file[path] = (text_norm, path_norm)

    for term in terms:
        t_norm = _normalize_lookup_token(term)
        if not t_norm:
            continue
        term_norm_cache[term] = t_norm
        hit_count = 0
        for text_norm, path_norm in norm_by_file.values():
            if t_norm in text_norm or t_norm in path_norm:
                hit_count += 1
        if hit_count > 0:
            df_by_term[t_norm] = hit_count

    target_norm = _normalize_lookup_token(target)
    exact_target_hit_files: set[str] = set()
    scored: list[tuple[str, float, list[str]]] = []

    for path, rec in file_records.items():
        text_norm, path_norm = norm_by_file[path]
        score = 0.0
        matched_terms: list[str] = []

        if target_norm:
            if target_norm in text_norm:
                score += 4.2
                exact_target_hit_files.add(path)
                matched_terms.append(target)
            if target_norm in path_norm:
                score += 5.0
                exact_target_hit_files.add(path)
                if target not in matched_terms:
                    matched_terms.append(target)

        for term in terms:
            t_norm = term_norm_cache.get(term) or _normalize_lookup_token(term)
            if not t_norm:
                continue

            hit_text = t_norm in text_norm
            hit_path = t_norm in path_norm
            if not hit_text and not hit_path:
                continue

            df = max(1, df_by_term.get(t_norm, 1))
            rarity = 1.0 / float(df)
            base = 0.8 + min(len(t_norm), 10) * 0.16
            if hit_text:
                score += base + rarity
            if hit_path:
                score += base * 1.15 + rarity
            if term not in matched_terms:
                matched_terms.append(term)

        if score > 0:
            scored.append((path, score, matched_terms))

    if not scored:
        return None

    scored.sort(key=lambda x: (-x[1], x[0]))

    if logger:
        top_debug = [f"{path}({score:.2f})" for path, score, _ in scored[:5]]
        logger.info(f"   \U0001f9ea [\u6587\u4ef6\u5b9a\u4f4d\u672c\u5730\u6253\u5206] target={target or '<none>'} | top={top_debug}")

    if len(exact_target_hit_files) == 1:
        best_path = next(iter(exact_target_hit_files))
        display_target = target or "该内容"
        return f"{display_target}在文件【{best_path}】中。"

    best_path, best_score, _ = scored[0]
    if len(scored) == 1:
        display_target = target or "该内容"
        return f"{display_target}在文件【{best_path}】中。"

    second_score = scored[1][1]
    if best_score >= second_score * 1.35 and (best_score - second_score) >= 1.2:
        display_target = target or "该内容"
        return f"{display_target}在文件【{best_path}】中。"

    lines = ["根据当前检索结果，可能涉及以下文件："]
    for i, (path, _, _) in enumerate(scored[: max(2, max_items)], start=1):
        lines.append(f"{i}. {path}")
    return "\n".join(lines)


def extract_related_topic(question: str) -> str:
    q = re.sub(r"\s+", "", (question or "").strip())
    if not q:
        return ""

    patterns = [
        r"(?:最近)?有哪些?和(.+?)(?:有关|相关)(?:的)?(?:记录|文档|文件)",
        r"(.+?)(?:有关|相关)(?:的)?(?:记录|文档|文件)有哪些?",
    ]
    for pattern in patterns:
        m = re.search(pattern, q)
        if not m:
            continue
        topic = (m.group(1) or "").strip("，。！？；：,.!?;:")
        topic = re.sub(r"^(和|与|跟|关于)", "", topic)
        return topic
    return ""


def _build_topic_variants(topic: str) -> list[str]:
    t = (topic or "").strip().lower()
    if not t:
        return []

    variants: list[str] = []

    def _add(x: str):
        s = (x or "").strip().lower()
        if s and s not in variants:
            variants.append(s)

    _add(t)

    for token in re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fa5]{2,}", t):
        _add(token)

    if re.fullmatch(r"[\u4e00-\u9fa5]{4,}", t):
        _add(t[:2])
        _add(t[-2:])
        for i in range(len(t) - 1):
            _add(t[i:i + 2])

    return variants


def maybe_build_related_records_answer(question: str, relevant_indices, repo_state, max_items: int = 8) -> str | None:
    if not is_related_record_listing_request(question):
        return None

    topic = extract_related_topic(question)
    if not topic:
        return None

    variants = _build_topic_variants(topic)
    if not variants:
        return None

    file_hits: dict[str, dict] = {}

    for idx in relevant_indices or []:
        path = repo_state.chunk_paths[idx]
        text = repo_state.chunk_texts[idx] or ""
        path_lower = path.lower()
        text_lower = text.lower()

        score = 0
        for v in variants:
            if v in path_lower:
                score = max(score, 3)
            if v in text_lower:
                score = max(score, 2)

        if score <= 0:
            continue

        evidence = ""
        for line in text.splitlines():
            ln = (line or "").strip()
            if not ln:
                continue
            if any(v in ln.lower() for v in variants):
                evidence = ln
                break

        rec = file_hits.get(path)
        dt = repo_state.chunk_file_times[idx]
        if not rec or score > rec["score"]:
            file_hits[path] = {
                "path": path,
                "score": score,
                "dt": dt,
                "evidence": evidence,
            }

    if not file_hits:
        return None

    items = [x for x in file_hits.values() if x["score"] >= 2]
    if not items:
        items = sorted(file_hits.values(), key=lambda x: x["score"], reverse=True)[:3]

    items = sorted(items, key=lambda x: x["dt"], reverse=True)[:max_items]

    lines = [f"根据现有记录，和“{topic}”有关的记录有 {len(items)} 条："]
    for i, item in enumerate(items, 1):
        path = item["path"]
        date_match = re.search(r"(20\d{2}-\d{2}-\d{2})", Path(path).name)
        date_text = date_match.group(1) if date_match else item["dt"].strftime("%Y-%m-%d")
        lines.append(f"{i}. {path}（{date_text}）")
        if item["evidence"]:
            ev = item["evidence"]
            if len(ev) > 80:
                ev = ev[:80] + "..."
            lines.append(f"   证据：{ev}")

    return "\n".join(lines)
