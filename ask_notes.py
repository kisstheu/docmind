import os
import time
import datetime
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

# ====== 1. 环境与启动计时 ======
start_init = time.time()
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

print("正在初始化系统...")
model_emb = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print(f"⏱️ 模型加载耗时: {time.time() - start_init:.2f}s")

NOTES_DIR = Path("test_notes")
CACHE_FILE = Path("brain_cache.npz")
MODEL_ID = "gemini-2.5-flash"

# 初始化 AI 客户端（提前初始化，供影子索引使用）
client = genai.Client(api_key=os.getenv("OPENAI_API_KEY"), vertexai=False)

# ====== 2. 读取文件与时间感知 ======
docs, paths = [], []
file_info_list = []

# 按文件的修改时间(st_mtime)从小到大排序
all_files = list(NOTES_DIR.glob("*"))
all_files.sort(key=lambda x: x.stat().st_mtime)

for file in all_files:
    if file.suffix.lower() not in {".txt", ".md"}: continue

    mtime = datetime.datetime.fromtimestamp(file.stat().st_mtime)
    date_str = mtime.strftime('%Y-%m-%d')

    docs.append(file.read_text(encoding="utf-8", errors="ignore"))
    paths.append(file.name)
    file_info_list.append(f"- {file.name} (更新于: {date_str})")

earliest_note = file_info_list[0] if file_info_list else "无"
latest_note = file_info_list[-1] if file_info_list else "无"

# ====== 3. 影子索引与向量缓存 ======
embeddings = None
if CACHE_FILE.exists():
    cache = np.load(CACHE_FILE, allow_pickle=True)
    if len(cache['paths']) == len(paths):
        embeddings = cache['embeddings']
        print("✨ 调取现成记忆缓存")

if embeddings is None:
    print("\n🧠 触发初次建库：正在启动 AI 引擎生成【影子索引】...")
    print("⏳ 注意：这将调用大模型为每篇笔记提取隐藏特征，可能需要几分钟，但只需执行一次！\n")

    enhanced_docs = []
    for path, doc in zip(paths, docs):
        print(f"   🤖 正在透视文件：{path} ...")
        try:
            # 逼迫 AI 提取深层实体和情绪标签
            summary_prompt = (
                f"请仔细阅读以下私人随手记片段，提取出 5-8 个最核心的搜索关键词。\n"
                f"要求：\n"
                f"1. 【向上推断核心项目】：不要只停留在字面技术词汇！根据你的世界知识，大胆推断并提取其所属的完整名。\n"
                f"2. 【提炼开发动作】：如果内容涉及代码修改、指针报错、依赖关系，自动补充‘编译调试’、‘源码剖析’等动作标签。\n"
                f"3. 极度简练，只返回用空格分隔的关键词，绝不解释。\n\n"
                f"文本：\n{doc[:2000]}"
            )
            resp = client.models.generate_content(model=MODEL_ID, contents=summary_prompt)
            shadow_tags = resp.text.strip()

            # 融合为增强版文本喂给本地向量模型
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
        "2. 当用户问‘有哪些笔记’、‘最早/最新笔记’等宏观问题时，直接参考【全局统计】和【全局地图】回答。\n"
        "3. 当用户问具体细节时，根据提供的【参考片段】回答，绝不瞎编。\n"
        "4. 语气自然、像真人一样聊天，极度简练。\n"
        "5. 【严格隔离与推理】：绝不能将不同工作/生活切片强行缝合。\n"
        "6. 【架构师思维与融会贯通】：当用户探讨某个宏观项目时，如果检索到的是底层的代码片段或报错记录，你要能像高级架构师一样，主动向用户解释这些底层代码是如何在整个项目的编译、重构和运行链路中发挥作用的。要自主建立代码细节与宏观项目之间的关联！"
    ),
    temperature=0.3
)
chat = client.chats.create(model=MODEL_ID, config=chat_config)

print(f"✅ 系统就绪！启动总耗时: {time.time() - start_init:.2f}s")
print("=================================")

# ====== 5. 带有短期记忆的对话循环 ======
memory_buffer = []

while True:
    question = input("\n问：")
    if question.strip().lower() in ['q', 'quit', 'exit']: break
    if not question.strip(): continue

    start_qa = time.time()

    try:
        greetings = ["你好", "嗨", "在吗", "谢谢", "好的", "ok", "嗯", "哈哈", "知道了", "原来如此"]

        if question in greetings:
            search_query = question
            print(f"🔍 [直觉模式]：纯寒暄，极速响应")
        else:
            history_str = "\n".join(memory_buffer[-4:])
            # 重写 Prompt，保留文件名等上下文线索
            rewrite_prompt = (
                f"【近期对话历史】\n{history_str}\n\n"
                f"【任务】\n"
                f"请结合上述历史，补全用户最新问题中缺失的上下文（主语或指代对象）。\n"
                f"特别注意：如果用户使用了代词或要求看“详细内容”，请务必将 AI 刚刚回答中提到的【文件名】或核心主题提取出来，拼接到关键词中。\n"
                f"将用户的最新问题‘{question}’重写为一个独立的、具体的搜索关键词短语。\n"
                f"【警告】：绝不允许捏造任何无关实体！直接返回关键词，不要任何解释。"
            )
            try:
                rewrite_resp = client.models.generate_content(model=MODEL_ID, contents=rewrite_prompt)
                search_query = rewrite_resp.text.strip()
                print(f"🔍 [意图重写]：{search_query}")
            except Exception as e:
                search_query = question
                print(f"⚠️ 重写失败，使用原句 ({e})")

        # 加入 BGE 中文短搜长的专属检索咒语
        bge_instruction = "为这个句子生成表示以用于检索相关文章："
        q_emb = model_emb.encode([bge_instruction + search_query])[0]
        scores = np.dot(embeddings, q_emb)

        temp_query = (question + " " + search_query).lower()
        exact_match_indices = []

        sorted_indices = sorted(range(len(paths)), key=lambda k: len(paths[k]), reverse=True)

        for i in sorted_indices:
            base_name = paths[i].replace(".txt", "").replace(".md", "").lower()
            if base_name and base_name in temp_query:
                exact_match_indices.append(i)
                print(f"⚡ [精确拦截]：检测到直接呼叫文件名 -> {paths[i]}")
                temp_query = temp_query.replace(base_name, " ")

        # 提高阈值，宁缺毋滥防串味
        threshold = 0.55
        relevant_indices = [i for i, s in enumerate(scores) if s > threshold]

        for idx in exact_match_indices:
            if idx not in relevant_indices:
                relevant_indices.insert(0, idx)

        if not relevant_indices and len(question) > 8:
            relevant_indices = [np.argsort(scores)[-1]]

        relevant_indices = relevant_indices[:4]  # 最多只给 4 个防爆炸

        context_text = ""
        if relevant_indices:
            # 塞给大模型的仍然是原始纯净文本 docs，不会干扰最终回答
            retrieved_docs = [f"文件【{paths[idx]}】：\n{docs[idx]}" for idx in relevant_indices]
            context_text = "【检索到的参考片段】:\n" + "\n---\n".join(retrieved_docs) + "\n\n"

        final_prompt = f"{context_text}【用户输入】: {question}"

        print(f"🔍 [系统日志] 匹配到 {len(relevant_indices)} 个相关片段...")
        response = chat.send_message(final_prompt)

        if response.text:
            print(f"\nAI回答：\n{response.text}")
            memory_buffer.append(f"用户：{question}")
            memory_buffer.append(f"AI：{response.text[:200]}...")

        print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")

    except Exception as e:
        print(f"\n调用失败: {e}")