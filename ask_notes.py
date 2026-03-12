import os
import time

# ====== 1. 环境与启动计时 ======
start_init = time.time()
os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

import numpy as np
from pathlib import Path
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

print("正在初始化系统...")
model_emb = SentenceTransformer("BAAI/bge-small-zh-v1.5")
print(f"⏱️ 模型加载耗时: {time.time() - start_init:.2f}s")

NOTES_DIR = Path("test_notes")
CACHE_FILE = Path("brain_cache.npz")
MODEL_ID = "gemini-2.5-flash"

# ====== 2. 读取文件与向量缓存 ======
docs, paths = [], []
for file in NOTES_DIR.glob("*"):
    if file.suffix.lower() not in {".txt", ".md"}: continue
    docs.append(file.read_text(encoding="utf-8", errors="ignore"))
    paths.append(file.name)

embeddings = None
if CACHE_FILE.exists():
    cache = np.load(CACHE_FILE, allow_pickle=True)
    if len(cache['paths']) == len(paths):
        embeddings = cache['embeddings']
        print("✨ 调取现成记忆缓存")

if embeddings is None:
    print("🧠 重新生成记忆向量...")
    embeddings = model_emb.encode(docs)
    np.savez(CACHE_FILE, embeddings=embeddings, paths=paths)

# ====== 3. 初始化带有“全局视野”的 AI ======
client = genai.Client(api_key=os.getenv("OPENAI_API_KEY"), vertexai=False)
file_map = "\n".join([f"- {p}" for p in paths])

chat_config = types.GenerateContentConfig(
    system_instruction=(
        "你是一个聪明、懂心思的个人笔记整理助手。你有‘全局地图’和‘局部细节’两层意识。\n"
        f"【全局地图】：你的仓库里目前有以下文件：\n{file_map}\n\n"
        "【行为原则】：\n"
        "1. 当用户只是寒暄（如“你好”）、道谢或输入很短时，直接自然回复，无需强行引用笔记。\n"
        "2. 当用户问‘有哪些笔记’等宏观问题时，利用【全局地图】概括。\n"
        "3. 当用户问具体细节时，根据提供的【参考片段】回答，绝不瞎编。\n"
        "4. 语气自然、像真人一样聊天，极度简练。"
    ),
    temperature=0.4
)
chat = client.chats.create(model=MODEL_ID, config=chat_config)

print(f"✅ 系统就绪！启动总耗时: {time.time() - start_init:.2f}s")
print("=================================")
# ====== 4. 对话循环 ======
# 自己维护“短期记忆缓冲区”
memory_buffer = []

while True:
    question = input("\n问：")
    if question.strip().lower() in ['q', 'quit', 'exit']: break
    if not question.strip(): continue

    start_qa = time.time()

    try:
        # A. 直觉模式 vs 意图重写
        if len(question) <= 8 or question in ["你好", "嗨", "在吗", "谢谢", "好的", "ok"]:
            search_query = question
            print(f"🔍 [直觉模式]：跳过重写，极速响应")
        else:
            # 取最近的两轮问答（最多 4 条记录）作为上下文提示
            history_str = "\n".join(memory_buffer[-4:])

            rewrite_prompt = (
                f"【近期对话历史】\n{history_str}\n\n"
                f"【任务】\n"
                f"请结合上述历史，理解用户最新问题中的代词（如‘这’、‘它’）具体指代什么。\n"
                f"将用户的最新问题‘{question}’重写为一个独立的、具体的搜索关键词短语。\n"
                f"直接返回关键词，不要任何解释。"
            )

            try:
                # 使用模型直接生成，避免污染对话对象
                rewrite_resp = client.models.generate_content(model=MODEL_ID, contents=rewrite_prompt)
                search_query = rewrite_resp.text.strip()
                print(f"🔍 [意图重写]：{search_query}")
            except Exception as e:
                search_query = question  # 容错兜底
                print(f"⚠️ 重写失败，使用原句 ({e})")

        # B. 检索阶段
        q_emb = model_emb.encode([search_query])[0]
        scores = np.dot(embeddings, q_emb)

        # 文件名精确拦截器 (Exact Match Interceptor)
        exact_match_indices = []
        for i, p in enumerate(paths):
            # 去掉后缀，只对比名字本身，提高命中率
            base_name = p.replace(".txt", "").replace(".md", "")
            if base_name in question or p in question:
                exact_match_indices.append(i)
                print(f"⚡ [精确拦截]：检测到直接呼叫文件名 -> {p}")

        # 动态阈值筛选
        threshold = 0.45
        relevant_indices = [i for i, s in enumerate(scores) if s > threshold]

        # 合并逻辑：将“精确拦截”的文件强行插入到最前面，无视阈值
        for idx in exact_match_indices:
            if idx not in relevant_indices:
                relevant_indices.insert(0, idx)

        # 只有在非寒暄、没命中向量、且没有精确拦截时，才触发“保底强塞 1 篇”机制
        if not relevant_indices and len(question) > 8:
            relevant_indices = [np.argsort(scores)[-1]]

        # 限制数量为 5，兼顾速度与准确度
        relevant_indices = relevant_indices[:5]

        # C. 组装与生成最终回答
        context_text = ""
        if relevant_indices:
            retrieved_docs = [f"文件【{paths[idx]}】：\n{docs[idx]}" for idx in relevant_indices]
            context_text = "【检索到的参考片段】:\n" + "\n---\n".join(retrieved_docs) + "\n\n"

        final_prompt = f"{context_text}【用户输入】: {question}"

        print(f"🔍 [系统日志] 匹配到 {len(relevant_indices)} 个相关片段...")
        response = chat.send_message(final_prompt)

        if response.text:
            print(f"\nAI回答：\n{response.text}")

            # 将成功的问答记录写入记忆缓冲区，供下一轮重写使用
            # 为了防止内存无限增长，甚至可以在这里做个截断，但通常列表存字符串消耗极小
            memory_buffer.append(f"用户：{question}")
            # 截取 AI 回答的前 200 个字，防止重写提示词爆炸
            memory_buffer.append(f"AI：{response.text[:200]}...")

        print(f"⏱️ 耗时: {time.time() - start_qa:.2f}s")

    except Exception as e:
        print(f"\n调用失败: {e}")