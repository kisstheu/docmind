from __future__ import annotations

import os
import re
import time
from pathlib import Path

from app.chat_loop import run_chat_loop
from bootstrap.env_setup import apply_environment_defaults
from infra.logging_setup import build_logger
from retrieval.repo_index import load_or_build_embeddings, scan_repository


def _slugify_path_name(path: Path) -> str:
    """
    将目录名转成适合做缓存目录/文件名的安全字符串。
    """
    name = path.name.strip() or "default"
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)


def _resolve_cache_file(notes_dir: Path, logger) -> Path:
    """
    为不同数据目录分配独立缓存文件，并兼容旧版 brain_cache.npz。

    兼容策略：
    1. 如果当前就是旧默认目录 test_notes，并且根目录下存在 brain_cache.npz，
       则继续直接使用它，避免重建。
    2. 否则使用 cache/<notes_dir_name>/brain_cache.npz
    """
    legacy_cache_file = Path("brain_cache.npz")
    legacy_default_notes_dir = Path("test_notes")

    if notes_dir.resolve() == legacy_default_notes_dir.resolve() and legacy_cache_file.exists():
        logger.info(f"🧠 检测到旧版缓存，继续复用: {legacy_cache_file}")
        return legacy_cache_file

    cache_dir = Path("cache") / _slugify_path_name(notes_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "brain_cache.npz"

    logger.info(f"🧠 当前数据目录独立缓存: {cache_file}")
    return cache_file


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

    notes_dir = Path("E:/test/kisstheu")
    cache_file = _resolve_cache_file(notes_dir, logger)

    model_id = "gemini-2.5-flash"
    ollama_api_url = "http://localhost:11434/api/generate"
    ollama_model = "qwen2.5"

    client = genai.Client(api_key=os.getenv("OPENAI_API_KEY"), vertexai=False)

    logger.info(f"📂 当前笔记目录: {notes_dir.resolve()}")
    logger.info(f"💾 当前缓存文件: {cache_file.resolve()}")

    scanned = scan_repository(notes_dir, logger)
    repo_state = load_or_build_embeddings(
        scanned,
        cache_file,
        model_emb,
        logger,
        ollama_api_url,
        ollama_model,
    )

    logger.info(f"✅ 系统就绪！启动总耗时: {time.time() - start_init:.2f}s")
    run_chat_loop(repo_state, model_emb, client, model_id, ollama_api_url, ollama_model, logger)


if __name__ == "__main__":
    main()