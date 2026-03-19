import sys
import os
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
import re
import time
import datetime
import docx2txt
import numpy as np
from pathlib import Path
import pymupdf
import requests
import torch
import logging
from google import genai
from google.genai import types
from rapidocr_onnxruntime import RapidOCR
from sentence_transformers import SentenceTransformer

# ====== 0. 配置企业级双轨日志系统 ======
logger = logging.getLogger("DocMind")
logger.setLevel(logging.DEBUG)  # 捕获所有级别的日志

# 1. 控制台输出
console_handler = logging.StreamHandler(sys.stdout)
# 💡 临时调试开关：把这里的 INFO 改成 DEBUG，终端就会把底层搜了什么词全打印出来！调试完再改回 INFO 即可。
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(console_handler)

# 2. 文件输出 (DEBUG级别：将所有底层打分和溯源细节写入文件，便于事后复盘)
file_handler = logging.FileHandler("docmind_sys.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s'))
logger.addHandler(file_handler)

# ====== 1. 环境与启动计时 ======
start_init = time.time()


logger.info("正在初始化系统...")

# 💡 硬件动态检测
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

try:
    model_emb = SentenceTransformer(
        "BAAI/bge-large-zh-v1.5",
        device=device,
        local_files_only=True
    )
except Exception as e:
    logger.error(f"⚠️ 本地嵌入模型加载失败：{e}")
    logger.error("请确认该模型已经提前下载到本机缓存，或改为使用本地模型目录。")
    raise

logger.info(f"⚙️ BGE 向量模型运行设备: {device.upper()}")
logger.info(f"⏱️ 模型加载耗时: {time.time() - start_init:.2f}s")

if not os.getenv("OPENAI_API_KEY"):
    logger.error("⚠️ [系统拦截] 未检测到大模型 API Key！请配置环境变量。")
    import sys
    sys.exit(1)

NOTES_DIR = Path("test_notes")
CACHE_FILE = Path("brain_cache.npz")
MODEL_ID = "gemini-2.5-flash"

# 本地大模型（用于快速提取意图、建库等脏活累活）
OLLAMA_API_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5"

client = genai.Client(api_key=os.getenv("OPENAI_API_KEY"), vertexai=False)

# ====== 2. 读取文件与时间感知 (全深度扫描) ======
docs, paths = [], []
file_times = []
file_info_list = []

chunk_texts = []
chunk_paths = []
chunk_meta = []
chunk_file_times = []
SUPPORTED_EXT = {".txt", ".md", ".docx", ".pdf"}
EXTENSION_TERMS = {"pdf", "docx", "txt", "md"}

raw_files = list(NOTES_DIR.rglob("*"))
all_files = []
for f in raw_files:
    if not f.is_file(): continue
    if any(part in f.parts for part in {'.venv', '.idea', '.git', '.SynologyWorkingDirectory', '__pycache__'}): continue
    if f.suffix.lower() not in SUPPORTED_EXT: continue
    if f.name.endswith(".ocr.txt") or f.name.startswith("~$"): continue
    all_files.append(f)

all_files.sort(key=lambda x: x.stat().st_mtime)


def robust_read_text(filepath):
    try:
        return filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = filepath.read_text(encoding="gb18030")
            logger.info(f"         💡 成功使用 GB18030(ANSI) 抢救出中文 -> {filepath.name}")
            return text
        except Exception:
            return filepath.read_text(encoding="utf-8", errors="ignore")

def extract_company_candidates(text: str):
    candidates = []

    patterns = [
        r'[\u4e00-\u9fa5A-Za-z0-9·（）()\-]{2,}股份有限公司',
        r'[\u4e00-\u9fa5A-Za-z0-9·（）()\-]{2,}有限公司',
        r'[\u4e00-\u9fa5A-Za-z0-9·（）()\-]{2,}集团',
        r'[\u4e00-\u9fa5A-Za-z0-9·（）()\-]{2,}公司',
    ]

    for _pattern in patterns:
        for m in re.findall(_pattern, text):
            name = m.strip("，。；：、（）() \n\t")
            if len(name) >= 2:
                candidates.append(name)

    # 去重保序
    _result = []
    seen = set()
    for name in candidates:
        if name not in seen:
            seen.add(name)
            _result.append(name)

    return _result

def classify_org_candidate(name: str):
    name = name.strip()

    # 明确的正式组织名
    if name.endswith("股份有限公司"):
        return "explicit"
    if name.endswith("有限公司"):
        return "explicit"
    if name.endswith("集团") and len(name) >= 4:
        return "explicit"

    # 明显泛称 / 指代
    generic_names = {
        "公司", "贵公司", "该公司", "本公司", "原公司",
        "大公司", "小公司", "对方公司"
    }
    if name in generic_names:
        return "generic"

    # 地点 + 公司 这类泛称，通常不算正式组织名
    if name.endswith("公司") and len(name) <= 6:
        return "generic"

    # 其他情况先视为模糊候选（简称、别称、未写全）
    return "ambiguous"


def extract_query_terms(_search_query: str, _question: str):
    text = f"{_search_query} {_question}"

    # 去掉标点
    text = re.sub(r"[^\w\s\u4e00-\u9fa5]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    raw_terms = []

    # 1. 保留英文/数字词
    raw_terms.extend(re.findall(r"[a-zA-Z0-9_]{2,}", text))

    # 2. 抓中文连续片段
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
        if len(term) < 2:
            continue
        if term in stop_terms:
            continue

        # 对“涉及了多少公司”这种句子做一点轻拆分
        if "公司" in term and term != "公司":
            cleaned.append("公司")
            continue
        if "人名" in term and term != "人名":
            cleaned.append("人名")
            continue
        if "项目" in term and term != "项目":
            cleaned.append("项目")
            continue

        cleaned.append(term)

    # 去重但保序
    _result = []
    for term in cleaned:
        if term not in _result:
            _result.append(term)

    return _result


def detect_inventory_target(_question: str):
    q = re.sub(r"[^\w\s\u4e00-\u9fa5]", "", _question)

    target_aliases = {
        "company": ["公司", "单位", "组织", "企业"],
        "person": ["人", "人名", "人物", "同事", "员工"],
        "project": ["项目", "系统", "方案"],
        "place": ["地点", "地方", "城市", "位置"],
    }

    for target_type, aliases in target_aliases.items():
        for alias in aliases:
            if alias in q:
                return target_type, alias

    return None, None

def chunk_text(text: str, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text = text.strip()

    if not text:
        return chunks

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end]

        chunks.append({
            "text": chunk,
            "start": start,
            "end": end
        })

        if end >= len(text):
            break

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def expand_neighbor_chunks(top_chunk_indices, chunk_paths, chunk_meta, neighbor=1):
    expanded = set()

    for idx in top_chunk_indices:
        expanded.add(idx)
        current_path = chunk_paths[idx]
        current_chunk_id = chunk_meta[idx]["chunk_id"]

        for j in range(max(0, idx - neighbor), min(len(chunk_paths), idx + neighbor + 1)):
            if chunk_paths[j] == current_path and abs(chunk_meta[j]["chunk_id"] - current_chunk_id) <= neighbor:
                expanded.add(j)

    return sorted(expanded)

for file in all_files:
    stat = file.stat()
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
    file_times.append(mtime)
    date_str = mtime.strftime('%Y-%m-%d')
    size_kb = stat.st_size / 1024
    content = ""

    try:
        if file.suffix.lower() in {".txt", ".md"}:
            content = robust_read_text(file)
        elif file.suffix.lower() == ".docx":
            try:
                content = docx2txt.process(file)
            except Exception:
                logger.info(f"      🕵️ 发现“伪装者”文件 {file.name}，正在尝试暴力读取...")
                content = robust_read_text(file)
        elif file.suffix.lower() == ".pdf":
            ocr_sidecar_path = file.with_name(file.name + ".ocr.txt")
            if ocr_sidecar_path.exists():
                logger.info(f"      📄 发现伴生文件，直接秒读纯文本 -> {ocr_sidecar_path.name}")
                content = robust_read_text(ocr_sidecar_path)
            else:
                ocr_engine = None
                with pymupdf.open(file) as pdf_doc:
                    for page_num, page in enumerate(pdf_doc):
                        page_text = page.get_text().strip()
                        if len(page_text) < 15:
                            if ocr_engine is None:
                                # 强制开启 GPU 加速
                                ocr_engine = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
                            logger.info(f"      👁️ 发现“金身”页面 (页码 {page_num + 1})，启动 OCR...")
                            pix = page.get_pixmap(dpi=150)
                            result, _ = ocr_engine(pix.tobytes("png"))
                            if result: page_text = "\n".join([line[1] for line in result])
                        content += page_text + "\n"
                if content.strip():
                    ocr_sidecar_path.write_text(content, encoding="utf-8")
    except Exception as e:
        logger.warning(f"⚠️ 解析文件失败 {file.name}: {e}")
        continue

    if content:
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = content.replace('\u200b', '').replace('\ufeff', '').strip()

    if not content: continue

    relative_path = file.relative_to(NOTES_DIR).as_posix()

    # 全文级信息
    docs.append(content)
    paths.append(relative_path)
    file_info_list.append(f"- {relative_path} (大小: {size_kb:.1f}KB, 更新于: {date_str})")

    # chunk级信息
    chunks = chunk_text(content, chunk_size=1000, overlap=200)

    for idx, ch in enumerate(chunks):
        chunk_texts.append(ch["text"])
        chunk_paths.append(relative_path)
        chunk_meta.append({
            "path": relative_path,
            "chunk_id": idx,
            "start": ch["start"],
            "end": ch["end"]
        })
        chunk_file_times.append(mtime)

earliest_note = file_info_list[0] if file_info_list else "无"
latest_note = file_info_list[-1] if file_info_list else "无"

# ====== 3. 影子索引与向量缓存 ======
embeddings = None
chunk_embeddings = None

if CACHE_FILE.exists():
    cache = np.load(CACHE_FILE, allow_pickle=True)
    if len(cache['paths']) == len(paths):
        embeddings = cache['embeddings']
        if all(k in cache.files for k in ['chunk_embeddings', 'chunk_texts', 'chunk_paths', 'chunk_meta', 'chunk_file_times']):
            chunk_embeddings = cache['chunk_embeddings']
            chunk_texts = list(cache['chunk_texts'])
            chunk_paths = list(cache['chunk_paths'])
            chunk_meta = list(cache['chunk_meta'])
            chunk_file_times = [datetime.datetime.fromtimestamp(float(ts)) for ts in cache['chunk_file_times']]
            logger.info("✨ 调取现成记忆缓存（含chunk索引）")
        else:
            logger.info("⚠️ 检测到旧版缓存，仅有全文索引，将重建 chunk 索引。")

if embeddings is None or chunk_embeddings is None:
    logger.info("\n🧠 触发初次建库：正在启动【本地 Ollama 引擎】生成影子索引...\n")
    enhanced_docs = []
    enhanced_chunk_texts = []

    for path, doc in zip(paths, docs):
        logger.info(f"   🤖 正在透视文件：{path} ...")
        try:
            summary_prompt = (
                f"请提取以下私人笔记片段的 5-8 个最核心搜索关键词。\n"
                f"【严格分类指令】：\n"
                f"1. [技术/职场类]：【仅当】内容明确涉及代码、软件开发、公司事务时，才允许加入“项目”、“工作”及技术词。\n"
                f"2. [游戏/娱乐类]：【仅当】明确涉及游戏时，才加入“游戏”、“个人爱好”及具体游戏名。\n"
                f"3. [生活类]：如果涉及生活琐事、情感日常，【绝对禁止】加入任何技术、代码或工作词汇！提取其本身的专属词即可。\n"
                f"【输出格式】：极度简练，只返回空格分隔的关键词，不许废话。\n\n"
                f"文本：\n{doc[:1500]}"
            )
            payload = {"model": OLLAMA_MODEL, "prompt": summary_prompt, "stream": False}
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
            response.raise_for_status()
            shadow_tags = response.json().get("response", "").strip()
            enhanced_docs.append(f"【核心隐藏特征：{shadow_tags}】\n{doc}")
            logger.info(f"      ✨ 提取到影子标签：[{shadow_tags}]")
        except Exception as e:
            logger.warning(f"      ⚠️ {path} 透视失败，使用原文本 ({e})")
            enhanced_docs.append(doc)

    for chunk_text_item, chunk_path, meta in zip(chunk_texts, chunk_paths, chunk_meta):
        enhanced_chunk_texts.append(
            f"【所属文件：{chunk_path}｜chunk:{meta['chunk_id']}】\n{chunk_text_item}"
        )

    logger.info("\n🧠 正在将全文级文档压入高维向量空间...")
    embeddings = model_emb.encode(enhanced_docs)

    logger.info("🧠 正在将 chunk 片段压入高维向量空间...")
    chunk_embeddings = model_emb.encode(enhanced_chunk_texts)

    np.savez(
        CACHE_FILE,
        embeddings=embeddings,
        paths=np.array(paths, dtype=object),
        chunk_embeddings=chunk_embeddings,
        chunk_texts=np.array(chunk_texts, dtype=object),
        chunk_paths=np.array(chunk_paths, dtype=object),
        chunk_meta=np.array(chunk_meta, dtype=object),
        chunk_file_times=np.array([dt.timestamp() for dt in chunk_file_times], dtype=float),
    )
    logger.info("✅ 影子索引建库完成！（含chunk索引）\n")

# ====== 4. 初始化 AI ======
chat_config = types.GenerateContentConfig(
    system_instruction=(
        "你是用户的个人笔记助理，擅长根据给定材料回答问题、整理线索、做有限归纳。\n"
        f"当前知识库共有 {len(paths)} 个文件。\n"
        f"时间跨度参考：最早文件是 {earliest_note}；最新文件是 {latest_note}。\n"
        "你只能依据本轮给出的参考片段回答，不要假设自己看过所有文件全文。\n"
        "有证据就答，没有证据就明确说信息不足。\n"
        "不要因为名字相似、简称相似，就擅自把不同人物、公司、项目混为一谈。\n"
    ),
    temperature=0.4
)

logger.info(f"✅ 系统就绪！启动总耗时: {time.time() - start_init:.2f}s")
print("=================================")
print("🤖：你好！我是你的 DocMind 随身助理。你可以问我任何问题。")

# ====== 5. 对话循环 ======
memory_buffer = []
current_focus_file = None


def answer_repo_meta_question(question: str, all_files, paths, file_times):
    q = question.lower()

    # 文件总数
    if any(k in question for k in ["文件总数", "多少个文件", "多少文件", "目前有多少文档"]):
        return f"目前共有 {len(all_files)} 个文件。"

    # 文件格式统计
    if any(k in question for k in ["多少格式", "几种格式", "有哪些格式", "文件格式"]):
        from collections import Counter
        ext_counter = Counter((f.suffix.lower() or "[无后缀]") for f in all_files)
        items = sorted(ext_counter.items(), key=lambda x: (-x[1], x[0]))
        detail = "，".join([f"{ext}: {count} 个" for ext, count in items])
        return f"当前共有 {len(ext_counter)} 种文件格式：{detail}。"

    # 最早文件
    if "最早" in question and ("笔记" in question or "文件" in question):
        if file_times:
            idx = min(range(len(file_times)), key=lambda i: file_times[i])
            return f"最早的文件是 {paths[idx]}，时间是 {file_times[idx].strftime('%Y-%m-%d %H:%M:%S')}。"

    # 最新文件
    if "最新" in question and ("笔记" in question or "文件" in question):
        if file_times:
            idx = max(range(len(file_times)), key=lambda i: file_times[i])
            return f"最新的文件是 {paths[idx]}，时间是 {file_times[idx].strftime('%Y-%m-%d %H:%M:%S')}。"

    return None


def answer_system_capability_question(question: str):
    if any(k in question for k in [
        "你是谁", "介绍一下"
    ]):
        return "我是你的 DocMind 随身助理，主要负责根据你的本地笔记和文档回答问题、整理线索，并做有限归纳。"

    if any(k in question for k in [
        "能干啥", "你能做什么", "能做什么", "可以做什么", "你可以做什么", "你的功能", "有什么功能", "有啥功能", "怎么用"
    ]):
        return (
            "我现在主要能做这几类事：\n"
            "1. 根据你的本地笔记、文档和资料回答具体问题。\n"
            "2. 帮你查某个人、某个项目、某段经历、某份材料里提到过什么。\n"
            "3. 帮你做有限总结，比如梳理某个方案、某组记录或某段时间线。\n"
            "4. 帮你做一些仓库层面的统计，比如文件数量、文件格式、最近更新情况等。\n"
            "5. 对于明显是闲聊、寒暄或系统介绍类问题，我也可以直接回答，不必走文档检索。"
        )

    return None


while True:
    question = input("\n问：")
    if question.strip().lower() in ['q', 'quit', 'exit']: break
    if not question.strip(): continue

    start_qa = time.time()
    if question.strip() in ["银", "仁", "人", "找仁", "找银"]:
        question = "找人"

    try:
        greetings = ["你好", "嗨", "在吗", "谢谢", "好的", "ok", "嗯", "哈哈", "知道了", "原来如此", "厉害", "棒",
                     "牛逼", "多谢", "感谢"]
        system_queries = [
            "能干啥", "你是谁", "怎么用", "你能做什么", "你的功能", "介绍一下",
            "能做什么", "可以做什么", "你可以做什么", "有什么功能", "有啥功能"
        ]
        repo_meta_queries = [
            "文件列表", "最早的笔记", "最新笔记",
            "文件总数", "多少个文件", "多少文件", "统计",
            "多少格式", "几种格式", "有哪些格式", "文件格式",
        ]
        local_system_answer = None
        if any(word in question for word in system_queries):
            local_system_answer = answer_system_capability_question(question)

        if local_system_answer:
            print(f"\nAI回答：\n{local_system_answer}")
            clean_reply = local_system_answer.replace("\n", " ").replace("*", "").replace("#", "")
            memory_buffer.append(f"用户：{question}")
            memory_buffer.append(f"AI：{clean_reply[:150] + ('...' if len(clean_reply) > 150 else '')}")
            print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")
            continue
        
        is_repo_meta_query = any(word in question for word in repo_meta_queries)
        if is_repo_meta_query:
            local_answer = answer_repo_meta_question(question, all_files, paths, file_times)
            if local_answer:
                print(f"\nAI回答：\n{local_answer}")
                clean_reply = local_answer.replace("\n", " ").replace("*", "").replace("#", "")
                memory_buffer.append(f"用户：{question}")
                memory_buffer.append(f"AI：{clean_reply[:150] + ('...' if len(clean_reply) > 150 else '')}")
                print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")
                continue
        inventory_triggers = ["多少", "哪些", "有哪些", "提到", "涉及", "所有", "盘点", "过"]
        inventory_target_type, inventory_target_label = detect_inventory_target(question)
        is_inventory_query = inventory_target_type is not None and any(t in question for t in inventory_triggers)
        relationship_queries = ["对我如何", "关系好", "评价", "他人怎么样", "对他"]

        skip_retrieval = False
        search_query = question
        is_relationship_query = any(q in question for q in relationship_queries)

        if any(word in question.lower() for word in greetings) and len(question) <= 15 and not is_relationship_query:
            skip_retrieval = True
        elif any(word in question for word in system_queries):
            skip_retrieval = True
        elif is_inventory_query:
            skip_retrieval = False
        else:
            history_str = "\n".join(memory_buffer[-4:])
            if not memory_buffer:
                task_desc = f"当前没有历史对话。请直接提取用户问题‘{question}’中的核心词作为搜索短语。绝对禁止脑补任何不存在的人名（如李四、张三）或项目名（如项目A）！"
            else:
                task_desc = (
                    f"请结合上述历史，理解用户最新问题‘{question}’的真实搜索意图。\n"
                    f"🚨【绝对禁止】：严禁直接复制或罗列历史对话中的实体列表（如一长串公司名、人名）！\n"
                    f"💡【正确做法】：将其抽象为3-5个高度概括的搜索关键词"
                )

            rewrite_prompt = (
                f"【近期对话历史】\n{history_str}\n\n"
                f"【任务】\n"
                f"{task_desc}\n"
                f"【最高警告】：\n"
                f"1. 绝对不允许脑补外部文件名！\n"
                f"2. 💡【转移话题判定】：如果用户暗示‘其他’、‘另外的’或‘从xxx出发’，必须立刻抛弃历史记录中的旧实体和旧文件名！\n"
                f"3. 🚨【实体保护原则】：如果用户最新提问中出现了明确的具体人名、地名或实体，重写后的搜索词【必须】包含该新实体，绝对不允许用历史记录中的旧名字去覆盖！\n"
                f"4. 🛑【禁止过度翻译】：如果用户输入了极短的英文字母，请【原封不动】地保留这些字母！绝对不允许脑补或翻译成词汇！也要防止将这些字母理解为文件后缀！\n"
                f"5. 🛑【输出格式绝对指令】：只能返回纯粹的词汇短语，词与词之间用空格隔开。绝对不允许输出完整的句子！绝对不允许包含问号、逗号、感叹号等任何标点符号！\n"
                f"6. 🕵️【侦探直觉与事实检索】：\n"
                f"   - 当用户询问‘人际关系’或‘都有谁’时，翻译为查找具体人名和事件。\n"
                f"7. 🚫【致命禁忌】：提取的搜索短语中【绝对不可以】包含“.txt”、“.md”、“.pdf”或“.docx”等扩展名！\n"
                f"8. 🗣️【上下文继承与纠错领悟】：如果用户输入极短（如‘A其实是B’），说明是在纠正上一轮的映射。你【必须】像人类一样，把上一轮的核心问题带入新的搜索词中！绝不能把核心意图弄丢！\n"
                f"9. 🤡【调侃与情绪过滤】：如果用户的最新提问带有表情包（如😄、😅、😂）或明显是随口调侃，请【仅保留】上一轮的核心实体（如人名）作为搜索词，绝对禁止脑补出“冲突”、“争执”、“暴力”等严肃词汇去污染搜索池！\n"
                f"10. 🏢【职场语义翻译】：如果用户询问某人的‘作品’、‘产出’、‘成果’或‘做过什么’，请务必将其翻译为具体的职场实体词汇，如：方案、文档、代码、系统、项目、设计。绝对不要只保留‘作品’这种偏文艺的词汇，避免导致技术文档检索失败！"
            )
            try:
                # 1. 组装发给本地 Ollama 的请求
                payload = {
                    "model": OLLAMA_MODEL,
                    "prompt": rewrite_prompt,
                    "stream": False
                }
                # 本地推理通常很快，设个 10 秒超时防卡死足够了
                response = requests.post(OLLAMA_API_URL, json=payload, timeout=10)
                response.raise_for_status()

                # 2. 获取本地小模型的提取结果
                search_query = response.json().get("response", "").strip()

                # 3. 必须先脱掉后缀！避免下一步的清洗把句号洗掉导致正则失效
                search_query = re.sub(r'\.(txt|md|pdf|docx)$', '', search_query, flags=re.IGNORECASE).strip()

                # 4. 物理清洗：移除非字母数字、非中文字符的各种标点符号（此时句号问号等都会被洗掉，只留干净的词）
                search_query = re.sub(r'[^\w\s\u4e00-\u9fa5]', ' ', search_query)

                # 5. 替换多个连续空格为一个空格，保持格式整洁
                search_query = re.sub(r'\s+', ' ', search_query).strip()

                logger.info(f"🔍 [本地引擎意图重写]：{search_query}")

            except Exception as e:
                logger.warning(f"⚠️ 本地意图提取失败，退回原问题：{e}")
                search_query = question

        # === 检索核心逻辑 ===
        scores, relevant_indices, exact_match_indices = [], [], []
        ignored_file = None

        if not skip_retrieval:
            q_emb = model_emb.encode(["为这个句子生成表示以用于检索相关文章：" + search_query])[0]
            scores = np.dot(chunk_embeddings, q_emb)
            apply_time_decay = not is_inventory_query
            if apply_time_decay:
                now = datetime.datetime.now()
                for i in range(len(scores)):
                    delta_days = (now - chunk_file_times[i]).days

                    if delta_days < 30:
                        time_weight = 1.0
                    elif delta_days < 180:
                        time_weight = 0.92
                    elif delta_days < 365:
                        time_weight = 0.85
                    else:
                        time_weight = 0.75

                    scores[i] *= time_weight

            search_terms = extract_query_terms(search_query, question)
            logger.debug(f"提取到的核心搜索词: {search_terms}")

            for i, chunk_text_item in enumerate(chunk_texts):
                path_no_ext = Path(chunk_paths[i]).stem.lower()

                for term in search_terms:
                    term_lower = term.lower()

                    if term_lower in path_no_ext:
                        if term_lower in EXTENSION_TERMS:
                            scores[i] += 0.02
                        else:
                            scores[i] += 0.25
                            logger.debug(f"   [chunk文件名命中] '{term}' -> {chunk_paths[i]}")

                    elif term_lower in chunk_text_item.lower():
                        if term_lower in EXTENSION_TERMS:
                            scores[i] += 0.02
                        elif re.match(r'^[a-z0-9_]+$', term_lower):
                            scores[i] += 0.25
                            logger.debug(f"   [chunk英文命中] '{term}' -> {chunk_paths[i]}")
                        else:
                            scores[i] += 0.12
                            logger.debug(f"   [chunk正文命中] '{term}' -> {chunk_paths[i]}")

            shift_keywords = ["其他", "别的", "所有", "全局", "抛开", "除了", "另外", "换个", "不说", "那"]
            if any(k in question for k in shift_keywords) or len(question) < 4:
                ignored_file = current_focus_file
                current_focus_file = None
                if ignored_file: logger.info(f"   🔄 [焦点释放] 检测到话题转移，临时屏蔽: {ignored_file}")

            temp_query = (question + " " + search_query).lower()
            sorted_indices = sorted(range(len(paths)), key=lambda k: len(paths[k]), reverse=True)

            for i in sorted_indices:
                full_name = paths[i].lower()
                base_name = Path(paths[i]).stem.lower()

                if ignored_file and full_name == ignored_file.lower():
                    continue

                if full_name in temp_query:
                    current_focus_file = paths[i]
                    temp_query = temp_query.replace(full_name, " ")
                    logger.info(f"   🎯 [精确拦截-全名] -> {paths[i]}")
                    break

                pattern = rf"(?:^|[^a-zA-Z0-9_]){re.escape(base_name)}(?:[^a-zA-Z0-9_]|$)"
                if not base_name.isdigit() and re.search(pattern, temp_query):
                    if temp_query.strip() == base_name or len(base_name) >= 4:
                        current_focus_file = paths[i]
                        temp_query = re.sub(pattern, " ", temp_query)
                        logger.info(f"   🎯 [精确拦截-词边界] -> {paths[i]}")
                        break

            if current_focus_file and not any(k in question for k in shift_keywords):
                logger.info(f"   🔒 [全局焦点锁定] AI注意力集中于 -> {current_focus_file}")
                for i in range(len(scores)):
                    if chunk_paths[i] == current_focus_file:
                        scores[i] += 0.18

            is_macro_request = is_inventory_query or any(
                kw in question for kw in
                [
                    "时间线", "经过", "梳理", "复盘", "总结", "详细", "过程",
                    "所有", "表现", "评价", "对吗", "境遇", "怎么看",
                    "经历", "待过"
                ]
            )
            is_person_eval_query = (
                    ("评价" in question or "怎么看" in question or "这个人怎么样" in question)
                    and any(len(term) >= 2 for term in extract_query_terms(search_query, question))
            )

            if is_person_eval_query:
                top_k = 6
                current_threshold = 0.50
            elif is_macro_request:
                top_k = 50
                current_threshold = 0.28
            else:
                top_k = 12
                current_threshold = 0.40

            if is_macro_request:
                logger.info(f"   📂 [深度核查模式] 上限扩至 {top_k} 份，及格线降至 {current_threshold}")

            semantic_indices = [i for i in np.argsort(scores)[::-1] if scores[i] > current_threshold]

            for idx in semantic_indices:
                if idx not in relevant_indices:
                    relevant_indices.append(idx)

            if not relevant_indices and len(search_query) >= 2:
                relevant_indices = [i for i in np.argsort(scores)[::-1] if scores[i] > 0.30]
                logger.info("   🛡️ [兜底打捞] 未达到阈值，启动底线捞取...")

            relevant_indices = relevant_indices[:top_k]
            logger.info(f"   🔍 [溯源完毕] 最终喂给大模型的片段数量: {len(relevant_indices)}")

        context_text = ""
        if relevant_indices:
            expanded_indices = expand_neighbor_chunks(
                top_chunk_indices=relevant_indices,
                chunk_paths=chunk_paths,
                chunk_meta=chunk_meta,
                neighbor=1
            )

            file_chunk_count = {}
            filtered_indices = []
            for idx in expanded_indices:
                p = chunk_paths[idx]
                file_chunk_count.setdefault(p, 0)
                if file_chunk_count[p] >= 3:
                    continue
                filtered_indices.append(idx)
                file_chunk_count[p] += 1

            context_blocks = []
            for idx in filtered_indices:
                meta = chunk_meta[idx]
                context_blocks.append(
                    f"文件【{chunk_paths[idx]}】（chunk #{meta['chunk_id']}，位置 {meta['start']}-{meta['end']}）：\n{chunk_texts[idx]}"
                )

            context_text = "【参考片段】:\n" + "\n---\n".join(context_blocks) + "\n\n"
            logger.debug(f"本次最终送入大模型的chunk文件列表: {list(dict.fromkeys([chunk_paths[idx] for idx in filtered_indices]))}")



        inventory_candidates_text = ""

        if is_inventory_query and inventory_target_type == "company":
            candidate_pool = []
            for doc_text in docs:
                for name in extract_company_candidates(doc_text):
                    candidate_pool.append(name)

            unique_names = []
            for name in candidate_pool:
                if name not in unique_names:
                    unique_names.append(name)

            deduped_names = []
            for name in unique_names:
                if any(name != other and name in other and len(other) >= len(name) + 2 for other in unique_names):
                    continue
                deduped_names.append(name)

            explicit_names = []
            ambiguous_names = []
            generic_names = []

            for name in deduped_names:
                name_type = classify_org_candidate(name)
                if name_type == "explicit":
                    explicit_names.append(name)
                elif name_type == "ambiguous":
                    ambiguous_names.append(name)
                else:
                    generic_names.append(name)

            def uniq_keep_order(items):
                _result = []
                for x in items:
                    if x not in _result:
                        _result.append(x)
                return _result

            explicit_names = uniq_keep_order(explicit_names)
            ambiguous_names = uniq_keep_order(ambiguous_names)
            generic_names = uniq_keep_order(generic_names)

            lines = []
            if explicit_names or ambiguous_names or generic_names:
                lines.append("【盘点候选：组织】")
                if explicit_names:
                    lines.append("【明确组织名】")
                    lines.extend([f"- {x}" for x in explicit_names[:20]])
                if ambiguous_names:
                    lines.append("【可能是简称或未写全】")
                    lines.extend([f"- {x}" for x in ambiguous_names[:20]])
                if generic_names:
                    lines.append("【泛称/指代】")
                    lines.extend([f"- {x}" for x in generic_names[:20]])

                inventory_candidates_text = "\n".join(lines) + "\n\n"

        focus_injection = (
            f"【当前焦点】当前对话优先围绕文件《{current_focus_file}》展开。"
            f"如果用户没有明确切换对象，请优先参考该文件及其相关片段，"
            f"不要随意扩展到其他无关文件或人物。\n"
            if current_focus_file else ""
        )

        final_prompt = (
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


        response = client.models.generate_content(model=MODEL_ID, contents=final_prompt, config=chat_config)

        if response.text:
            print(f"\nAI回答：\n{response.text}")
            clean_reply = response.text.replace("\n", " ").replace("*", "").replace("#", "")
            memory_buffer.append(f"用户：{question}")
            memory_buffer.append(f"AI：{clean_reply[:150] + ('...' if len(clean_reply) > 150 else '')}")

        print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")

    except Exception as e:
        logger.error(f"\n调用失败: {e}")