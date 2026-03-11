import os

from google.genai import types

os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

from google import genai
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# ====== 配置 ======
NOTES_DIR = Path("test_notes")

client = genai.Client(
    api_key=os.getenv("OPENAI_API_KEY"),
    vertexai=False
)

MODEL_ID = "gemini-2.5-flash"

model_emb = SentenceTransformer("all-MiniLM-L6-v2")

# ====== 读取文件 ======
docs = []
paths = []
if not NOTES_DIR.exists():
    print(f"错误：找不到目录 {NOTES_DIR}")
    exit()

for file in NOTES_DIR.glob("*"):
    if file.suffix.lower() not in {".txt", ".md"}: continue
    docs.append(file.read_text(encoding="utf-8", errors="ignore"))
    paths.append(file.name)

print(f"读取 {len(docs)} 个文档")
if len(docs) == 0:
    print("未读到文档，请检查路径。")
    exit()

embeddings = model_emb.encode(docs)


# ====== 提问与持续对话 ======

# 1. 在循环外面，创建一个有记忆的“聊天对象”
chat = client.chats.create(model=MODEL_ID)

# 1. 给 AI 定制“灵魂”和“规矩”
chat_config = types.GenerateContentConfig(
    system_instruction=(
        "你是一个聪明、懂用户心思的个人笔记整理助手。"
        "你的任务是【理解用户的核心意图】，而不是像复读机一样机械地提取或翻译笔记。"
        "原则："
        "1. 极度简练：当用户要求总结方向、方面或分类时，只给高度抽象的核心词（如：技术开发、个人灵修），绝不要罗列里面的细节！"
        "2. 像真人一样聊天：语气自然、接地气，不要总是冷冰冰地列举1234。"
        "3. 惜字如金：用户不问细节，你绝对不说细节。"
    ),
    temperature=0.4 # 稍微调低一点温度，让它的回答更理智、聚焦
)

# 2. 用带规矩的配置创建聊天对象
chat = client.chats.create(
    model=MODEL_ID,
    config=chat_config
)

print("\n=================================")
print("🧠 注入灵魂的 AI 助手已就绪！可以开始提问了。(输入 q 退出)")
print("=================================")

while True:
    question = input("\n问：")
    if question.strip().lower() in ['q', 'quit', 'exit']:
        print("再见！👋")
        break
    if not question.strip():
        continue

    q_emb = model_emb.encode([question])[0]
    scores = np.dot(embeddings, q_emb)
    top_idx = np.argsort(scores)[-3:][::-1]

    context = ""
    for i in range(len(docs)):
        context += f"\n[{paths[i]}]\n{docs[i]}\n"

    # 把笔记内容偷偷塞进问题里
    prompt = f"相关参考笔记：\n{context}\n\n请结合以上笔记和我之前的对话，回答：{question}"

    try:
        response = chat.send_message(prompt)
        print("\nAI回答：")
        if response.text:
            print(response.text)
    except Exception as e:
        print(f"\n调用 Gemini 失败，错误详情: \n{e}")