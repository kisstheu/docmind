from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 加载本地小模型（测绘员）
print("正在加载本地小模型 (all-MiniLM-L6-v2)...")
# model_emb = SentenceTransformer("all-MiniLM-L6-v2")
# 专门为中文优化的小模型
model_emb = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 2. 读取笔记
NOTES_DIR = Path("test_notes")
docs = []
paths = []
for file in NOTES_DIR.glob("*"):
    if file.suffix.lower() not in {".txt", ".md"}: continue
    docs.append(file.read_text(encoding="utf-8", errors="ignore"))
    paths.append(file.name)

print(f"成功读取 {len(docs)} 个文档，正在将它们向量化...")
# 3. 把所有笔记变成“数学坐标”（建库）
embeddings = model_emb.encode(docs)
print("✅ 向量库建立完成！\n")

# 4. 体验检索效果
while True:
    question = input("你想找什么笔记？(输入 q 退出): ")
    if question.strip().lower() == 'q':
        break

    # 把问题变成坐标
    q_emb = model_emb.encode([question])[0]

    # 算距离（微型 Milvus）
    scores = np.dot(embeddings, q_emb)

    # 找出得分最高的 3 篇
    top_idx = np.argsort(scores)[-3:][::-1]

    print("\n🔍 找到了最相关的 3 篇笔记：")
    for i, idx in enumerate(top_idx):
        print(f"Top {i + 1}: {paths[idx]} (相似度得分: {scores[idx]:.4f})")
    print("-" * 50, "\n")