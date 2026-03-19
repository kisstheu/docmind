from __future__ import annotations

import os
import time
from pathlib import Path

from app.chat_loop import run_chat_loop
from bootstrap.env_setup import apply_environment_defaults
from infra.logging_setup import build_logger
from retrieval.repo_index import load_or_build_embeddings, scan_repository


def main():
    apply_environment_defaults()
    logger = build_logger()
    start_init = time.time()
    logger.info("正在初始化系统...")

    import_start = time.time()
    import torch
    from sentence_transformers import SentenceTransformer

    logger.info(f"📦 模型相关库导入耗时: {time.time() - import_start:.2f}s")

    model_load_start = time.time()
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    try:
        model_emb = SentenceTransformer("BAAI/bge-large-zh-v1.5", device=device, local_files_only=True)
    except Exception as e:
        logger.error(f"⚠️ 本地嵌入模型加载失败：{e}")
        logger.error("请确认该模型已经提前下载到本机缓存，或改为使用本地模型目录。")
        raise

    logger.info(f"⚙️ BGE 向量模型运行设备: {device.upper()}")
    logger.info(f"⏱️ 模型加载耗时: {time.time() - model_load_start:.2f}s")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("⚠️ [系统拦截] 未检测到大模型 API Key！请配置环境变量。")
        raise SystemExit(1)

    from google import genai

    notes_dir = Path("test_notes")
    cache_file = Path("brain_cache.npz")
    model_id = "gemini-2.5-flash"
    ollama_api_url = "http://localhost:11434/api/generate"
    ollama_model = "qwen2.5"

    client = genai.Client(api_key=os.getenv("OPENAI_API_KEY"), vertexai=False)
    scanned = scan_repository(notes_dir, logger)
    repo_state = load_or_build_embeddings(scanned, cache_file, model_emb, logger, ollama_api_url, ollama_model)

    logger.info(f"✅ 系统就绪！启动总耗时: {time.time() - start_init:.2f}s")
    run_chat_loop(repo_state, model_emb, client, model_id, ollama_api_url, ollama_model, logger)


if __name__ == "__main__":
    main()
