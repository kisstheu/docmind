import os

os.environ["http_proxy"] = "http://127.0.0.1:7897"
os.environ["https_proxy"] = "http://127.0.0.1:7897"

from google import genai

# 初始化客户端
client = genai.Client(
    api_key=os.getenv("OPENAI_API_KEY"),
    vertexai=False,
)

print("正在向 Google 请求可用模型列表...\n")

try:
    # 调用获取模型列表的方法
    models = client.models.list()

    count = 0
    for m in models:
        # 直接打印模型名称，不再做属性过滤
        print(f"模型名称: {m.name}")
        count += 1

    print(f"\n=================================")
    print(f"共找到 {count} 个模型！")

    if count == 0:
        print("警告：获取到了列表，但没有任何模型。这通常意味着账号没有分配配额或受限于地区。")

except Exception as e:
    print(f"\n获取模型列表彻底失败，错误详情:\n{e}")