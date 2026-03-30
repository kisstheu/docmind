from __future__ import annotations

import re
from typing import Optional, Tuple

EXTENSION_TERMS = {
    "pdf", "doc", "docx", "txt", "md",
    "xls", "xlsx", "csv",
    "ppt", "pptx",
    "png", "jpg", "jpeg", "bmp", "webp",
    "word", "excel", "ppt",
    "image",
}

SUPPORTED_EXT = {
    ".txt", ".md", ".doc", ".docx", ".pdf",
    ".xls", ".xlsx", ".csv",
    ".ppt", ".pptx",
    ".png", ".jpg", ".jpeg", ".bmp", ".webp",
}


ORG_NOISE_FRAGMENTS = {
    "关于", "根据", "要求", "通过", "提供", "未能", "人员", "身份", "权限",
    "劳动", "关系", "沟通", "确认", "通知", "因为", "由于",
    "本人", "我司", "我们", "该公司", "贵公司", "本公司", "对方公司",
}


def _looks_like_org_noise(name: str) -> bool:
    n = (name or "").strip()
    if not n:
        return True

    if any(frag in n for frag in ORG_NOISE_FRAGMENTS):
        return True

    if re.search(r"(并且|以及|或者|如果|是否|已经|仍然)", n):
        return True

    if len(n) > 40:
        return True

    return False


def extract_company_candidates(text: str):
    if not text:
        return []

    candidates = []
    patterns = [
        r'[\u4e00-\u9fa5A-Za-z0-9·（）()\-]{2,}股份有限公司',
        r'[\u4e00-\u9fa5A-Za-z0-9·（）()\-]{2,}有限公司',
        r'[\u4e00-\u9fa5A-Za-z0-9·（）()\-]{2,}集团',
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            match = m.group(0)
            name = match.strip("，。；：、（）() \n\t")
            if len(name) >= 2:
                if _looks_like_org_noise(name):
                    continue
                candidates.append(name)

    title_patterns = [
        (
            r'([\u4e00-\u9fa5A-Za-z0-9·（）()\-]{2,30})\s*[·•]\s*'
            r'(?:高级|资深|首席|招聘|HR|人事|经理|总监|负责人|创始人|CEO|CTO|COO|CFO|VP)'
        ),
    ]
    for pattern in title_patterns:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            raw = (m.group(1) or "").strip("，。；：、（）() \n\t")
            if len(raw) < 2:
                continue
            if re.search(r"(先生|女士|老师|同学)$", raw):
                continue
            if _looks_like_org_noise(raw):
                continue
            candidates.append(raw)

    result = []
    seen = set()
    for name in candidates:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def classify_org_candidate(name: str):
    name = name.strip()
    if _looks_like_org_noise(name):
        return "generic"

    if name.endswith("股份有限公司"):
        return "explicit"
    if name.endswith("有限公司"):
        return "explicit"
    if name.endswith("集团") and len(name) >= 4:
        return "explicit"

    generic_names = {
        "公司", "贵公司", "该公司", "本公司", "原公司",
        "大公司", "小公司", "对方公司"
    }
    if name in generic_names:
        return "generic"
    if name.endswith("公司") and len(name) <= 6:
        return "generic"
    if len(name) <= 2:
        return "generic"
    return "ambiguous"


def normalize_extension_term(term: str) -> str:
    t = (term or "").strip().lower()

    alias_map = {
        "word": "word",
        "word文档": "word",
        "doc": "doc",
        "docx": "docx",
        "pdf": "pdf",
        "txt": "txt",
        "md": "md",
        "markdown": "md",
        "excel": "excel",
        "xls": "xls",
        "xlsx": "xlsx",
        "csv": "csv",
        "ppt": "ppt",
        "pptx": "pptx",
        "png": "png",
        "jpg": "jpg",
        "jpeg": "jpeg",
        "bmp": "bmp",
        "webp": "webp",
        "图片": "image",
        "图像": "image",
        "image": "image",
    }
    return alias_map.get(t, t)


def extract_query_terms(search_query: str, question: str):
    text = f"{search_query} {question}"
    text = re.sub(r"[^\w\s\u4e00-\u9fa5]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    raw_terms = []
    raw_terms.extend(re.findall(r"[a-zA-Z0-9_]{2,}", text))
    raw_terms.extend(re.findall(r"[\u4e00-\u9fa5]{2,}", text))

    stop_terms = {
        "我想", "想知道", "请问", "一下", "这个", "那个", "这里", "那里",
        "涉及", "涉及了", "提到", "提到了", "多少", "几个", "哪些", "所有",
        "有没有", "是什么", "什么", "怎么", "为什么", "吗", "呢", "呀", "啊",
        "的呢", "过的", "我的", "那就"
    }

    cleaned = []
    for term in raw_terms:
        term = term.strip()
        if len(term) < 2 or term in stop_terms:
            continue

        normalized = normalize_extension_term(term)

        if normalized in EXTENSION_TERMS:
            cleaned.append(normalized)
            continue

        if "公司名称" in term or "公司名" in term:
            cleaned.append("公司")
            cleaned.append("名称")
            continue

        if "企业名称" in term or "单位名称" in term or "组织名称" in term:
            cleaned.append("名称")
            continue

        # 追问阶段模型可能把“公司名/公司名称”改写成“公司信息”，这里回补名称锚词。
        if "公司信息" in term or "企业信息" in term or "单位信息" in term or "组织信息" in term:
            cleaned.append("公司")
            cleaned.append("名称")
            continue

        if "公司" in term and term != "公司":
            cleaned.append("公司")
            if any(x in term for x in ("名称", "名字", "名")):
                cleaned.append("名称")
            continue
        if "人名" in term and term != "人名":
            cleaned.append("人名")
            continue
        if "项目" in term and term != "项目":
            cleaned.append("项目")
            continue
        cleaned.append(term)

    result = []
    for term in cleaned:
        if term not in result:
            result.append(term)
    return result


def detect_inventory_target(question: str) -> Tuple[Optional[str], Optional[str]]:
    q = re.sub(r"[^\w\s\u4e00-\u9fa5]", "", question)

    # 先判断更具体的文档格式类目标
    doc_aliases = [
        "word",
        "word文档",
        "doc",
        "docx",
        "pdf",
        "txt",
        "md",
        "excel",
        "xls",
        "xlsx",
        "csv",
        "ppt",
        "pptx",
        "png",
        "jpg",
        "jpeg",
        "bmp",
        "webp",
        "image",
        "图片",
        "图像",
    ]
    for alias in doc_aliases:
        if alias in q.lower():
            return "document", alias

    target_aliases = {
        "company": ["公司", "单位", "组织", "企业"],
        "person": ["人", "人名", "人物", "同事", "员工"],
        "project": ["项目", "系统", "方案"],
        "place": ["地点", "地方", "城市", "位置"],
        "document": ["文件", "文档", "资料", "word文档", "pdf文档"],
    }
    for target_type, aliases in target_aliases.items():
        for alias in aliases:
            if alias in q:
                return target_type, alias
    return None, None
