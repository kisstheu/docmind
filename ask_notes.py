import os
import re
import time
import datetime

import docx2txt
import numpy as np
from pathlib import Path

import pymupdf
import requests
from google import genai
from google.genai import types
from rapidocr_onnxruntime import RapidOCR
from sentence_transformers import SentenceTransformer

# ====== 1. 环境与启动计时 ======
start_init = time.time()
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print("正在初始化系统...")
model_emb = SentenceTransformer("BAAI/bge-large-zh-v1.5")
print(f"⏱️ 模型加载耗时: {time.time() - start_init:.2f}s")

NOTES_DIR = Path("test_notes")
CACHE_FILE = Path("brain_cache.npz")
MODEL_ID = "gemini-2.5-flash"

# 初始化 AI 客户端（提前初始化，供影子索引使用）
client = genai.Client(api_key=os.getenv("OPENAI_API_KEY"), vertexai=False)

# ====== 2. 读取文件与时间感知 ======
docs, paths = [], []
file_info_list = []

# 支持的格式列表
SUPPORTED_EXT = {".txt", ".md", ".docx", ".pdf"}

# 按文件的修改时间(st_mtime)从小到大排序
all_files = list(NOTES_DIR.glob("*"))
all_files.sort(key=lambda x: x.stat().st_mtime)

for file in all_files:
    if file.suffix.lower() not in SUPPORTED_EXT:
        continue
    if file.name.endswith(".ocr.txt"):
        continue
    # 1. 获取文件元数据 (大小和时间)
    stat = file.stat()
    mtime = datetime.datetime.fromtimestamp(stat.st_mtime)
    date_str = mtime.strftime('%Y-%m-%d')
    size_kb = stat.st_size / 1024

    # 2. 智能提取文本与 OCR 破甲（引入伴生文件机制）
    content = ""
    try:
        if file.suffix.lower() in {".txt", ".md"}:
            content = file.read_text(encoding="utf-8", errors="ignore")

        elif file.suffix.lower() == ".docx":
            content = docx2txt.process(file)

        elif file.suffix.lower() == ".pdf":
            # 检查是否存在 OCR 伴生文件
            ocr_sidecar_path = file.with_name(file.name + ".ocr.txt")

            if ocr_sidecar_path.exists():
                print(f"      📄 发现伴生文件，直接秒读纯文本 -> {ocr_sidecar_path.name}")
                content = ocr_sidecar_path.read_text(encoding="utf-8", errors="ignore")
            else:
                # 原来的 PDF 正常提取和 OCR 破甲逻辑
                ocr_engine = None

                with pymupdf.open(file) as pdf_doc:
                    for page_num, page in enumerate(pdf_doc):
                        page_text = page.get_text().strip()

                        if len(page_text) < 15:
                            if ocr_engine is None:
                                ocr_engine = RapidOCR()
                            print(f"      👁️ 发现“金身”页面 (页码 {page_num + 1})，启动 OCR 视觉强行破拆...")
                            pix = page.get_pixmap(dpi=150)
                            img_bytes = pix.tobytes("png")

                            result, _ = ocr_engine(img_bytes)
                            if result:
                                page_text = "\n".join([line[1] for line in result])
                                print(f"         ✅ 破拆成功！提取到 {len(page_text)} 个字符。")
                            else:
                                print(f"         ❌ 破拆失败，页面可能真的是空白的。")

                        content += page_text + "\n"

                # 破拆完成后，保存为伴生文件，下次一劳永逸！
                if content.strip():
                    ocr_sidecar_path.write_text(content, encoding="utf-8")

    except Exception as e:
        print(f"⚠️ 解析文件失败 {file.name}: {e}")
        continue

    # 3. 拦截空文件或全是图片的 PDF
    if not content.strip():
        print(f"⚠️ 文件 {file.name} 提取为空，已跳过")
        continue

    # 4. 存入系统记忆
    docs.append(content)
    paths.append(file.name)
    file_info_list.append(f"- {file.name} (大小: {size_kb:.1f}KB, 更新于: {date_str})")

earliest_note = file_info_list[0] if file_info_list else "无"
latest_note = file_info_list[-1] if file_info_list else "无"

# ====== 3. 影子索引与向量缓存 (Ollama 本地版) ======
embeddings = None
if CACHE_FILE.exists():
    cache = np.load(CACHE_FILE, allow_pickle=True)
    if len(cache['paths']) == len(paths):
        embeddings = cache['embeddings']
        print("✨ 调取现成记忆缓存")

if embeddings is None:
    print("\n🧠 触发初次建库：正在启动【本地 Ollama 引擎】生成影子索引...")
    print("⏳ 注意：这将调用你本地的 GPU 进行脱机推断，数据绝对安全！\n")

    enhanced_docs = []
    # 配置本地 Ollama 的 API 地址和使用的模型
    OLLAMA_API_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "qwen2.5"

    for path, doc in zip(paths, docs):
        print(f"   🤖 正在透视文件：{path} ...")
        try:
            # 强化版影子索引指令：明确边界
            summary_prompt = (
                f"请提取以下私人笔记片段的 5-8 个最核心搜索关键词。\n"
                f"【严格分类指令】：\n"
                f"1. [技术/职场类]：【仅当】内容明确涉及代码、软件开发、公司事务时，才允许加入“项目”、“工作”及技术词。\n"
                f"2. [游戏/娱乐类]：【仅当】明确涉及游戏时，才加入“游戏”、“个人爱好”及具体游戏名。\n"
                f"3. [生活类]：如果涉及生活琐事、情感日常，【绝对禁止】加入任何技术、代码或工作词汇！提取其本身的专属词即可。\n"
                f"【输出格式】：极度简练，只返回空格分隔的关键词，不许废话。\n\n"
                f"文本：\n{doc[:1500]}"
            )

            payload = {
                "model": OLLAMA_MODEL,
                "prompt": summary_prompt,
                "stream": False
            }

            # 向本地发送请求
            response = requests.post(OLLAMA_API_URL, json=payload, timeout=180)
            response.raise_for_status()

            shadow_tags = response.json().get("response", "").strip()

            enhanced_text = f"【核心隐藏特征：{shadow_tags}】\n{doc}"
            enhanced_docs.append(enhanced_text)
            print(f"      ✨ 提取到影子标签：[{shadow_tags}]")

        except Exception as e:
            print(f"      ⚠️ {path} 透视失败，使用原文本 ({e})")
            enhanced_docs.append(doc)

    print("\n🧠 正在将带有影子索引的笔记压入高维向量空间...")
    embeddings = model_emb.encode(enhanced_docs)
    np.savez(CACHE_FILE, embeddings=embeddings, paths=paths)
    print("✅ 影子索引建库完成！\n")


# ====== 4. 初始化带有“全局视野+时间维度”的 AI ======
file_map = "\n".join(file_info_list)

chat_config = types.GenerateContentConfig(
    system_instruction=(
        "你是一个聪明、懂心思的个人笔记整理助手。你有‘全局地图’和‘局部细节’两层意识。\n"
        f"【全局统计】：你的仓库里目前共有 {len(paths)} 个文件。\n"
        f"⏳ 时间跨度：最早的笔记是 {earliest_note}；最新更新的笔记是 {latest_note}。\n"
        f"【全局地图】（以下文件已按时间从早到晚排序）：\n{file_map}\n\n"
        "【行为原则】：\n"
        "1. 当用户只是寒暄、道谢或输入很短时，直接自然回复，无需强行引用笔记。\n"
        "2. 当用户问‘有哪些笔记’、‘最早/最新笔记’、‘提到了哪些人’等宏观统计问题时，请直接参考【全局统计】和上面的【全局地图】文件名进行推理回答。\n"
        "3. 当用户问具体细节时，优先根据提供的【参考片段】回答。\n"
        "4. 【自我认知】：当用户问“你能干啥”、“你是谁”等关于你系统能力的问题时，请直接自我介绍，告诉用户你可以帮他们精准检索、总结和推理海量的私人随手记，无需强行引用或分析具体的笔记片段。\n" 
        "5. 语气自然、像真人一样聊天，极度简练。\n"
        "6. 【严格隔离与推理】：绝不能将不同工作/生活切片强行缝合。必须明确区分用户的【个人娱乐项目】与【公司企业工作项目】，绝不可混淆！\n"
        "7. 【架构师思维与融会贯通】：当用户探讨某个宏观项目时，你要能主动建立代码细节与宏观项目的关联。"
    ),
    temperature=0.4
)

print(f"✅ 系统就绪！启动总耗时: {time.time() - start_init:.2f}s")
print("=================================")

print("🤖：你好！我是你的 DocMind 随身助理。")
print("    我已加载完所有笔记，你可以问我关于项目、生活的任何问题。")

# ====== 5. 带有短期记忆的无状态对话循环 ======
memory_buffer = []
current_focus_file = None  # 用于记录当前的全局焦点文件

while True:
    question = input("\n问：")
    if question.strip().lower() in ['q', 'quit', 'exit']: break
    if not question.strip(): continue

    start_qa = time.time()

    # 物理外挂：极简方言与错别字强制纠音
    if question.strip() in ["银", "仁", "人", "找仁", "找银"]:
        print("💡 [系统纠音]：检测到极简方言或错字，已强制翻译为 -> 找人")
        question = "找人"

    try:
        # 意图分类短路器
        greetings = ["你好", "嗨", "在吗", "谢谢", "好的", "ok", "嗯", "哈哈", "知道了", "原来如此", "厉害",
                     "棒", "牛逼", "多谢", "感谢"]
        system_queries = ["能干啥", "你是谁", "怎么用", "你能做什么", "你的功能", "介绍一下"]
        macro_queries = ["有哪些人", "提到哪些人", "所有人名", "文件列表", "最早的笔记", "最新笔记", "文件总数",
                         "多少个文件", "查人", "找人", "找个人", "多少文件", "统计"]

        skip_retrieval = False
        search_query = question

        # 如果是问人际关系或评价，禁止走 [直觉模式]
        relationship_queries = ["对我如何", "关系好", "评价", "他人怎么样", "对他"]
        is_relationship_query = any(q in question for q in relationship_queries)

        if any(word in question.lower() for word in greetings) and len(question) <= 15 and not is_relationship_query:
            print(f"🔍 [直觉模式]：检测到寒暄或夸奖，极速响应")
            skip_retrieval = True
        elif any(word in question for word in system_queries):
            print(f"🔍 [系统认知模式]：用户在问我能干啥，跳过文档检索")
            skip_retrieval = True
        elif any(word in question for word in macro_queries):
            print(f"🔍 [宏观统计模式]：用户在问全局信息，直接让 AI 查阅【全局地图】")
            skip_retrieval = True
        else:
            history_str = "\n".join(memory_buffer[-4:])
            rewrite_prompt = (
                f"【近期对话历史】\n{history_str}\n\n"
                f"【任务】\n"
                f"请结合上述历史，补全用户最新问题中缺失的上下文。\n"
                f"将用户的最新问题‘{question}’重写为一个独立的、具体的搜索关键词短语。\n"
                f"【最高警告】：\n"
                f"1. 如果近期对话历史为空，必须只根据当前提问提取关键词，绝对不允许脑补外部文件名！\n"
                f"2. 💡【转移话题判定】：如果用户暗示‘其他’、‘另外的’或‘从xxx出发’，必须立刻抛弃历史记录中的旧实体和旧文件名！\n"
                f"3. 🚨【实体保护原则】：如果用户最新提问中出现了明确的具体人名、地名或实体，重写后的搜索词【必须】包含该新实体，绝对不允许用历史记录中的旧名字去覆盖！\n"
                f"4. 🛑【禁止过度翻译】：如果用户输入了极短的英文字母，请【原封不动】地保留这些字母！绝对不允许脑补或翻译成词汇！也要防止将这些字母理解为文件后缀！\n"
                f"5. 只能返回纯粹的搜索短语，绝不允许输出多余的解释。\n"
                f"6. 🕵️【侦探直觉】：如果用户试图‘寻找背后实体’，请从历史上下文中提取技术凭证（如用户名）加入搜索词！\n"
                f"7. 🚫【致命禁忌】：提取的搜索短语中【绝对不可以】包含“.txt”或“.md”等扩展名！\n"
                f"8. 🗣️【方言与纠错领悟】：如果用户使用了方言谐音，或者用户在纠正你上一轮的错误，请务必像个人类一样领会其真正的意图，将搜索词纠正为标准普通话！"
            )
            try:
                rewrite_resp = client.models.generate_content(model=MODEL_ID, contents=rewrite_prompt)
                search_query = rewrite_resp.text.strip()
                # 暴力清洗大模型可能违规生成的后缀
                search_query = search_query.replace(".txt", "").replace(".md", "")
                print(f"🔍 [意图重写]：{search_query}")
            except Exception as e:
                search_query = question
                print(f"⚠️ 重写失败，回退使用原句作为查询词 ({e})")

        # ==========================================
        # 🚀 检索核心逻辑 (受短路器控制)
        # ==========================================
        scores = []
        relevant_indices = []
        exact_match_indices = []
        ignored_file = None

        if not skip_retrieval:
            # 加入 BGE 中文短搜长的专属检索咒语
            bge_instruction = "为这个句子生成表示以用于检索相关文章："
            q_emb = model_emb.encode([bge_instruction + search_query])[0]
            scores = np.dot(embeddings, q_emb)

            # ------------------------------------------
            # 🚀 关键词混合暴击 (Keyword Boost) + 智能降维打击
            # ------------------------------------------
            search_terms = [term for term in search_query.split() if len(term) >= 2]
            raw_eng_terms = re.findall(r'[a-zA-Z0-9_]{2,}', question)
            search_terms.extend(raw_eng_terms)
            search_terms = list(set(search_terms))

            # 智能词频侦测，判断哪些是“泛滥词”
            term_in_filename_count = {term: 0 for term in search_terms}
            for path in paths:
                path_no_ext = path.rsplit('.', 1)[0].lower()
                for term in search_terms:
                    if term.lower() in path_no_ext:
                        term_in_filename_count[term] += 1

            for i, doc_text in enumerate(docs):
                for term in search_terms:
                    term_lower = term.lower()
                    path_no_ext = paths[i].rsplit('.', 1)[0].lower()
                    doc_lower = doc_text.lower()

                    if term_lower in path_no_ext:
                        if term_in_filename_count[term] > 3:
                            scores[i] += 0.15
                            print(f"      🔥 [泛滥词降级] '{term}' 命中文件名，按普通权重加分 -> {paths[i]}")
                        else:
                            scores[i] += 0.35
                            print(f"      🔥 [文件名暴击] '{term}' 强力锁定文件 -> {paths[i]}")

                    elif term_lower in doc_lower:
                        # 纯英文/拼音特权！如果是纯字母或数字组合，给予 0.35 巨额保送分！
                        if re.match(r'^[a-z0-9_]+$', term_lower):
                            scores[i] += 0.35
                            print(f"      🔥 [英文特权暴击] '{term}' 强行捞出文件 -> {paths[i]}")
                        else:
                            scores[i] += 0.15
                            print(f"      🔥 [正文暴击] '{term}' 命中文件 -> {paths[i]}")


            # ------------------------------------------
            # 智能焦点释放
            # ------------------------------------------
            shift_keywords = ["其他", "别的", "所有", "全局", "抛开", "除了", "另外", "换个", "不说", "那"]

            # 如果问题极短，或者明确带有转移词，释放焦点
            if any(keyword in question for keyword in shift_keywords) or len(question) < 4:
                ignored_file = current_focus_file  # 把当前焦点打入冷宫
                current_focus_file = None
                print(f"🔄 [焦点释放]：检测到话题可能转移，已自动解除全局焦点锁定！(临时屏蔽: {ignored_file})")

            # ------------------------------------------
            # 混合检索逻辑 (Merge Strategy)
            # ------------------------------------------
            temp_query = (question + " " + search_query).lower()
            sorted_indices = sorted(range(len(paths)), key=lambda k: len(paths[k]), reverse=True)

            # 1. 先抓取精确匹配的文件
            for i in sorted_indices:
                full_name = paths[i].lower()
                base_name = full_name.replace(".txt", "").replace(".md", "")

                # 【严格执行黑名单机制！如果该文件被标记为忽略，直接跳过本次匹配
                if ignored_file and full_name == ignored_file.lower():
                    continue

                # 策略A：全名命中（带后缀），最精准，直接拦截
                if full_name in temp_query:
                    exact_match_indices.append(i)
                    print(f"⚡ [精确拦截-全名]：-> {paths[i]}")
                    temp_query = temp_query.replace(full_name, " ")
                    continue

                # 策略B优化：兼容中英文混合的“抗劫持”边界匹配
                pattern = rf"(?:^|[^a-zA-Z0-9_]){re.escape(base_name)}(?:[^a-zA-Z0-9_]|$)"

                if not base_name.isdigit() and re.search(pattern, temp_query):
                    if temp_query.strip() == base_name or len(base_name) >= 4:
                        exact_match_indices.append(i)
                        print(f"⚡ [精确拦截-词边界匹配]：-> {paths[i]}")
                        temp_query = re.sub(pattern, " ", temp_query)

            # 2. 更新焦点（如果用户没有喊“其他”，且命中了具体文件，就锁定焦点）
            if exact_match_indices and not any(k in question for k in shift_keywords):
                current_focus_file = paths[exact_match_indices[0]]
                print(f"🎯 [全局焦点锁定]：AI的注意力已死死盯住 -> {current_focus_file}")

            # 3. 获取 BGE 语义检索的高分结果
            threshold = 0.48
            semantic_indices = [i for i in np.argsort(scores)[::-1] if scores[i] > threshold]

            # 4. 将 精确命中 与 语义命中 优雅合并，去重
            for idx in exact_match_indices:
                if idx not in relevant_indices:
                    relevant_indices.append(idx)
            for idx in semantic_indices:
                if idx not in relevant_indices:
                    relevant_indices.append(idx)

            # 宁缺毋滥
            if not relevant_indices and len(search_query) >= 2:
                best_match_idx = np.argsort(scores)[-1]
                if scores[best_match_idx] > 0.35:  # 增加底线，低于此值绝对不给
                    relevant_indices = [best_match_idx]

            # 截断：最多只给 4 个文件
            relevant_indices = relevant_indices[:4]
            print(f"🔍 [系统日志] 匹配到 {len(relevant_indices)} 个相关片段...")

        context_text = ""
        if relevant_indices:
            retrieved_docs = [f"文件【{paths[idx]}】：\n{docs[idx]}" for idx in relevant_indices]
            context_text = "【本次检索到的独家参考片段】:\n" + "\n---\n".join(retrieved_docs) + "\n\n"

        # 1. 提取干净的历史记录
        clean_history = "\n".join(memory_buffer[-4:])

        # 2. 焦点注入文本
        focus_injection = f"【当前全局焦点文件】：{current_focus_file} (如果用户使用代词或省略主语，请务必默认围绕此文件展开！)\n" if current_focus_file else ""

        # 3. 组装无状态的专属 Prompt
        final_prompt = (
            f"【近期聊天上下文】:\n{clean_history}\n\n"
            f"{focus_injection}"
            f"{context_text}"
            f"【用户最新提问】: {question}\n\n"
            f"【最高指令】：\n"
            f"1. 如果【本次检索到的独家参考片段】里有内容，请优先分析片段。\n"
            f"2. 💡【全局联想限制】：你可以根据【全局地图/文件列表】回答“有哪些笔记”或“文件有多大”。但是！如果用户问及某个文件的【具体内容/它记录了什么】，你【必须】依赖下方提供的参考片段！如果参考片段中没有该文件的内容，【绝对禁止】望文生义或根据文件名瞎猜，必须诚实回答“未检索到该文件的具体内容”。\n" 
            f"3. 优先结合【当前全局焦点文件】的背景来理解用户的追问（如“这个”、“这家”等）。\n"
            f"4. 😈 启动‘毒舌模式’：针对技术/工作项目，如果内容包含抱怨，请犀利点评其拉胯之处，不要端着！\n"
            f"5. 🚨【个人项目保护】：明确区分游戏与工作。\n"
            f"6. ⏰【绝对时间线警告】：严格区分“笔记记录的时间（文件修改时间）”和“笔记内容涉及的技术年代”。\n"
            f"   - 如果用户问“当时/写这篇笔记时”，请【务必】以检索片段头部标注的 [修改时间: xxxx-xx-xx] 为准！\n"
            f"   - 切勿因为笔记里提到了老旧技术，就臆断用户是在那个年代写的笔记！\n"
            f"7. 🕵️【极客侦探模式】：在分析文档关联时，请高度敏锐地捕捉用户名、邮箱前缀、账号等技术凭证！\n"
            f"   - 💡 特别注意：如果用户提到极短的拼音首字母缩写，请【优先】将其理解为人名缩写或账号名，而【不是】文件扩展名！\n"
            f"   - 利用这些线索跨文档推理人物身份或项目背景，大胆猜测！\n"
            f"8. 🛑【实体隔离警告】：严禁跨年份、跨事件强行拼接实体（公司名、人名）！\n"
            f"   - 如果当前检索的文件没有具体名称，请直接回答‘找不到全称’。\n"
            f"   - 绝对不允许把文档里的公司名强行套用到，不要自己编造剧情！\n"
            f"9. 🛡️【拒绝胡诌指令】：如果检索到的片段内容与用户的问题【完全无关】（例如用户问宏观背景，检索到的全是具体代码日志），请勇敢地回答‘我的笔记里没有记录这方面的信息’，绝对禁止生搬硬套！\n"
            f"10. ⏳【时间线侧写与变化推理】：当用户提问关于时间跨度内的‘变化’、‘成长’或‘不同’时：\n"
            f"    - 【必须】立刻审视你系统指令中自带的【全局地图/文件列表】！\n"
            f"    - 观察从最早年份到最新年份的文件命名规律。你的关注点、使用的技术栈、参与的项目类型是否发生了转移？\n"
            f"    - 结合本次检索到的历史片段作为佐证，基于以上客观事实，为用户梳理出一条清晰的‘演进轨迹’。\n"
            f"    - 如果单靠片段不够，就大胆地用【全局地图】里的文件名来补充说明你的发现！"
        )

        # 使用 generate_content 保证每次独立思考，不污染长期记忆
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=final_prompt,
            config=chat_config
        )

        if response.text:
            print(f"\nAI回答：\n{response.text}")

            # 清洗掉换行和复杂的 Markdown 符号，避免下一轮语法错乱
            clean_reply = response.text.replace("\n", " ").replace("*", "").replace("#", "")
            # 找到前 150 个字符里最后一个句号/标点的位置进行优雅截断
            short_reply = clean_reply[:150]
            if len(clean_reply) > 150:
                short_reply += "..."

            memory_buffer.append(f"用户：{question}")
            memory_buffer.append(f"AI：{short_reply}")

        print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")

    except Exception as e:
        print(f"\n调用失败: {e}")