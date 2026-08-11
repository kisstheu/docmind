from __future__ import annotations

from dataclasses import dataclass
import re


MINIMUM_JD_CHARACTERS = 80

TITLE_LABELS = (
    "岗位名称",
    "职位名称",
    "招聘岗位",
    "招聘职位",
)
REQUIREMENT_LABELS = (
    "任职要求",
    "岗位要求",
    "职位要求",
    "技能要求",
    "技术要求",
    "任职资格",
    "岗位资格",
)
DUTY_LABELS = (
    "岗位职责",
    "职位职责",
    "工作职责",
    "职责描述",
    "岗位描述",
    "职位描述",
    "工作内容",
)
FIELD_LABELS = {
    "薪资": ("薪资范围", "薪酬范围", "薪资", "薪酬"),
    "工作地点": ("工作地点", "办公地点"),
    "工作制": ("工作制度", "工作制"),
    "加班或大小周": ("加班或大小周", "加班情况", "大小周", "加班"),
    "外包": ("是否外包", "外包"),
    "驻场": ("是否驻场", "驻场"),
    "学历": ("学历要求", "最低学历", "学历"),
    "经验": ("工作经验", "经验要求", "经验"),
}

_STRUCTURAL_LABELS = (
    *DUTY_LABELS,
    "公司介绍",
    "福利待遇",
    "职位福利",
)
_ALL_LABELS = tuple(
    sorted(
        {
            *TITLE_LABELS,
            *REQUIREMENT_LABELS,
            *_STRUCTURAL_LABELS,
            *(label for labels in FIELD_LABELS.values() for label in labels),
        },
        key=len,
        reverse=True,
    )
)
_LABEL_PATTERN = re.compile(
    rf"(?P<label>{'|'.join(re.escape(label) for label in _ALL_LABELS)})\s*[:：]"
)
_SEPARATORS = str.maketrans("", "", "-_/／·•|｜—–")
_ALLOWED_INSTRUCTION = re.compile(
    r"(?:请)?(?:帮我)?\s*(?:整理|提取|列出)(?:一下)?\s*"
    r"(?:这份|以下|下面|当前)?\s*(?:招聘\s*)?(?:JD|岗位|职位)?\s*"
    r"(?:中|里|的)?\s*(?:明确(?:写出|说明)?的?)?\s*(?:岗位)?(?:约束|条件|要求|信息)"
    r"[。.]?",
    re.IGNORECASE,
)
_REJECTED_INTENT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"值不值得",
        r"适不适合(?:我|本人)?",
        r"是否适合(?:我|本人)",
        r"结合.{0,20}(?:简历|偏好|历史)",
        r"根据.{0,20}(?:简历|偏好|历史)",
        r"(?:比较|对比|排序).{0,20}(?:岗位|职位|JD)",
        r"(?:岗位|职位|JD).{0,20}(?:比较|对比|排序)",
        r"(?:调查|查询|了解).{0,12}公司(?:背景)?",
        r"公司(?:背景)?.{0,12}(?:调查|查询)",
        r"(?:投递|面试|薪资谈判|职业规划).{0,10}建议",
        r"(?:分析|推荐|评价|判断).{0,20}(?:岗位|职位|JD|薪资)",
        r"(?:分析|评价|判断|列出|识别|看看).{0,12}(?:风险|红旗|好坏)",
        r"(?:岗位|职位|JD).{0,12}(?:风险|红旗|好坏)",
        r"薪资.{0,10}(?:建议|谈判)",
        r"(?:匹配|对照).{0,20}(?:简历|偏好)",
        r"(?:简历|偏好).{0,20}(?:匹配|对照)",
        r"(?:我的|个人|用户).{0,8}(?:简历|偏好)",
        r"(?:文件|文档|Source|检索结果|历史对话).{0,20}(?:补充|结合|读取|提取)",
        r"(?:补充|结合|读取).{0,20}(?:文件|文档|Source|检索结果|历史对话)",
        r"(?:请|帮我).{0,8}(?:总结|概括|改写|润色)",
        r"(?:https?://|www\.)",
        r"(?:打开|访问|查询).{0,20}(?:网页|网站|链接)",
        r"(?:还需要|请补充|需要追问).{0,20}(?:信息|内容)",
    )
)
_OCR_SALARY_LINE = re.compile(
    r"^[)）\]】]?\s*(?P<value>\d+(?:\.\d+)?\s*(?:[kK千])?\s*"
    r"[-–—~～至到]\s*\d+(?:\.\d+)?\s*[kK千](?:元)?"
    r"(?:\s*/?\s*(?:月|每月))?)\s*$"
)
_OCR_EXPERIENCE_RANGE = re.compile(
    r"(?<![\d.])(?P<value>\d+(?:\.\d+)?\s*[-–—~～至到]\s*"
    r"\d+(?:\.\d+)?\s*年)"
)
_OCR_TITLE_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9+#]*(?:[._/-][A-Za-z0-9+#]+)*"
)
_OCR_EXCLUDED_SECTION = re.compile(
    r"^\s*(?:个人简历|求职意向|教育经历|项目经历|工作经历|"
    r"合同条款|采购需求|采购规格|供应商资格|招标要求|交付要求|项目需求)"
    r"\s*[:：]?\s*$",
    re.MULTILINE,
)
_OCR_TRAILING_BOUNDARY = re.compile(
    r"^[^\r\n]*(?:本周活跃)\s*$|^\s*(?:去App|工作地址)\s*[:：]?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_OCR_TITLE_FORBIDDEN = re.compile(r"[:：；;。！？?]")
_OCR_MAX_HEADER_LINES = 16
_OCR_MIN_TITLE_CHARACTERS = 2
_OCR_MAX_TITLE_CHARACTERS = 48


@dataclass(frozen=True)
class LabelOccurrence:
    label: str
    start: int
    value_start: int


@dataclass(frozen=True)
class StructuralMatch:
    text: str
    occurrences: tuple[LabelOccurrence, ...]
    title_occurrences: tuple[LabelOccurrence, ...]
    requirement_occurrences: tuple[LabelOccurrence, ...]
    title_values: tuple[str, ...]
    shape: str = "canonical"
    pre_extracted_fields: tuple[tuple[str, str], ...] = ()
    requirement_content_ends: tuple[int, ...] = ()


def normalize_title(value: str) -> str:
    collapsed = "".join(value.strip().casefold().split())
    return collapsed.translate(_SEPARATORS)


def find_label_occurrences(text: str) -> tuple[LabelOccurrence, ...]:
    return tuple(
        LabelOccurrence(
            label=match.group("label"),
            start=match.start(),
            value_start=match.end(),
        )
        for match in _LABEL_PATTERN.finditer(text)
    )


def value_after(
    text: str,
    occurrence: LabelOccurrence,
    occurrences: tuple[LabelOccurrence, ...],
) -> str:
    end = len(text)
    for candidate in occurrences:
        if candidate.start > occurrence.start:
            end = candidate.start
            break
    return text[occurrence.value_start:end].strip(" \t\r\n-—:：;；,，。")


def _strip_approved_instruction(query: str) -> tuple[str, bool]:
    text = query.strip()
    instruction_count = 0
    first_label = _LABEL_PATTERN.search(text)
    if first_label is not None:
        prefix = text[: first_label.start()].strip()
        if prefix:
            if _ALLOWED_INSTRUCTION.fullmatch(prefix) is None:
                return text, False
            text = text[first_label.start() :].lstrip()
            instruction_count += 1

    suffix_match = re.search(
        rf"(?:^|\s)({_ALLOWED_INSTRUCTION.pattern})\s*$",
        text,
        re.IGNORECASE,
    )
    if suffix_match is not None and suffix_match.start(1) > 0:
        text = text[: suffix_match.start()].rstrip(" \t\r\n。.")
        instruction_count += 1
    return text, instruction_count <= 1


def _has_rejected_intent(query: str) -> bool:
    return any(pattern.search(query) for pattern in _REJECTED_INTENT_PATTERNS)


def _count_complete_blocks(
    title_occurrences: tuple[LabelOccurrence, ...],
    requirement_occurrences: tuple[LabelOccurrence, ...],
    text_length: int,
) -> int:
    blocks = 0
    for index, title in enumerate(title_occurrences):
        end = (
            title_occurrences[index + 1].start
            if index + 1 < len(title_occurrences)
            else text_length
        )
        if any(title.start < requirement.start < end for requirement in requirement_occurrences):
            blocks += 1
    return blocks


def _nonempty_lines(text: str) -> tuple[tuple[str, int], ...]:
    lines: list[tuple[str, int]] = []
    for match in re.finditer(r"[^\r\n]+", text):
        value = match.group().strip()
        if value:
            lines.append((value, match.start()))
    return tuple(lines)


def _has_repeated_ocr_title_token(title: str, body: str) -> bool:
    tokens = {
        match.group().casefold()
        for match in _OCR_TITLE_TOKEN.finditer(title)
        if len("".join(character for character in match.group() if character.isalnum()))
        >= 2
    }
    body_folded = body.casefold()
    return any(token in body_folded for token in tokens)


def _recognize_canonical(query: str) -> StructuralMatch | None:
    text, instruction_is_valid = _strip_approved_instruction(query)
    if not instruction_is_valid:
        return None
    if "?" in text or "？" in text:
        return None
    if sum(not character.isspace() for character in text) < MINIMUM_JD_CHARACTERS:
        return None

    occurrences = find_label_occurrences(text)
    title_occurrences = tuple(item for item in occurrences if item.label in TITLE_LABELS)
    requirement_occurrences = tuple(
        item for item in occurrences if item.label in REQUIREMENT_LABELS
    )
    if not title_occurrences or not requirement_occurrences:
        return None

    title_values = tuple(
        value_after(text, item, occurrences) for item in title_occurrences
    )
    normalized_titles = {normalize_title(value) for value in title_values if value}
    if not normalized_titles or len(normalized_titles) != 1:
        return None
    if any(not value for value in title_values):
        return None
    if _count_complete_blocks(title_occurrences, requirement_occurrences, len(text)) != 1:
        return None

    return StructuralMatch(
        text=text,
        occurrences=occurrences,
        title_occurrences=title_occurrences,
        requirement_occurrences=requirement_occurrences,
        title_values=title_values,
    )


def _recognize_ocr(query: str) -> StructuralMatch | None:
    text = query.strip()
    if "?" in text or "？" in text:
        return None
    if sum(not character.isspace() for character in text) < MINIMUM_JD_CHARACTERS:
        return None
    if _OCR_EXCLUDED_SECTION.search(text) is not None:
        return None
    lines = _nonempty_lines(text)
    if any(_ALLOWED_INSTRUCTION.fullmatch(line) is not None for line, _ in lines):
        return None

    occurrences = find_label_occurrences(text)
    if any(item.label in TITLE_LABELS for item in occurrences):
        return None
    duty_occurrences = tuple(item for item in occurrences if item.label in DUTY_LABELS)
    requirement_occurrences = tuple(
        item for item in occurrences if item.label in REQUIREMENT_LABELS
    )
    if len(duty_occurrences) != 1 or len(requirement_occurrences) != 1:
        return None
    duty = duty_occurrences[0]
    requirement = requirement_occurrences[0]
    if duty.start >= requirement.start:
        return None

    header_lines = tuple(
        item for item in lines if item[1] < duty.start
    )
    if len(header_lines) < 3 or len(header_lines) > _OCR_MAX_HEADER_LINES:
        return None
    title = header_lines[0][0]
    title_length = sum(not character.isspace() for character in title)
    if not (_OCR_MIN_TITLE_CHARACTERS <= title_length <= _OCR_MAX_TITLE_CHARACTERS):
        return None
    if _OCR_TITLE_FORBIDDEN.search(title) is not None:
        return None

    salary_matches = tuple(
        (index, match.group("value").strip())
        for index, (line, _) in enumerate(header_lines)
        if (match := _OCR_SALARY_LINE.fullmatch(line)) is not None
    )
    if len(salary_matches) != 1 or salary_matches[0][0] != 1:
        return None
    experience_values = tuple(
        match.group("value").strip()
        for line, _ in header_lines
        for match in _OCR_EXPERIENCE_RANGE.finditer(line)
    )
    if len(experience_values) != 1:
        return None

    trailing = _OCR_TRAILING_BOUNDARY.search(text, requirement.value_start)
    requirement_end = trailing.start() if trailing is not None else len(text)
    if requirement_end <= requirement.value_start:
        return None
    if not _has_repeated_ocr_title_token(title, text[duty.start:requirement_end]):
        return None

    return StructuralMatch(
        text=text,
        occurrences=occurrences,
        title_occurrences=(),
        requirement_occurrences=requirement_occurrences,
        title_values=(title,),
        shape="ocr",
        pre_extracted_fields=(
            ("薪资", salary_matches[0][1]),
            ("经验", experience_values[0]),
        ),
        requirement_content_ends=(requirement_end,),
    )


def recognize_query(query: str) -> StructuralMatch | None:
    if _has_rejected_intent(query):
        return None
    canonical = _recognize_canonical(query)
    if canonical is not None:
        return canonical
    return _recognize_ocr(query)
