from __future__ import annotations

import argparse
import hashlib
import os
import re
import time
from pathlib import Path

from app.chat_loop import run_chat_loop
from bootstrap.env_setup import apply_environment_defaults
from infra.debug_question_trace import build_debug_question_recorder
from infra.logging_setup import build_logger
from retrieval.repo_index import load_or_build_embeddings, scan_repository


def _slugify_path_name(path: Path) -> str:
    """
    将目录名转成适合做缓存目录/文件名的安全字符串。
    """
    name = path.name.strip() or "default"
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    path_hash = hashlib.sha1(str(path.resolve()).lower().encode("utf-8")).hexdigest()[:8]
    return f"{safe_name}_{path_hash}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DocMind local notes chat.")
    parser.add_argument(
        "-n",
        "--notes-dir",
        default=None,
        help="Path to notes directory. Overrides DOCMIND_NOTES_DIR.",
    )
    return parser.parse_args()


def _resolve_notes_dir(args: argparse.Namespace, logger) -> Path:
    cli_value = (args.notes_dir or "").strip()
    env_value = (os.getenv("DOCMIND_NOTES_DIR") or "").strip()

    if cli_value:
        source = "--notes-dir"
        raw_path = cli_value
    elif env_value:
        source = "DOCMIND_NOTES_DIR"
        raw_path = env_value
    else:
        source = "default"
        raw_path = "examples/demo_notes_public"

    notes_dir = Path(raw_path).expanduser()
    if not notes_dir.is_absolute():
        notes_dir = (Path.cwd() / notes_dir).resolve()
    else:
        notes_dir = notes_dir.resolve()

    if not notes_dir.exists():
        logger.error(f"❌ 笔记目录不存在（来源: {source}）: {notes_dir}")
        raise SystemExit(2)
    if not notes_dir.is_dir():
        logger.error(f"❌ 笔记目录不是目录（来源: {source}）: {notes_dir}")
        raise SystemExit(2)

    logger.info(f"📁 笔记目录来源: {source}")
    return notes_dir


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


def _resolve_embedding_local_only() -> bool:
    raw = (os.getenv("DOCMIND_EMBEDDING_LOCAL_ONLY") or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return False


def _resolve_change_log_file(cache_file: Path, logger) -> Path:
    change_log_file = cache_file.parent / "file_change_log.db"
    logger.info(f"🧾 文件变更日志库: {change_log_file}")
    return change_log_file


def main():
    args = _parse_args()
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

    embedding_local_only = _resolve_embedding_local_only()
    try:
        model_emb = SentenceTransformer(
            "BAAI/bge-large-zh-v1.5",
            device=device,
            local_files_only=embedding_local_only,
        )
    except Exception as e:
        mode_label = "本地缓存模式" if embedding_local_only else "在线下载模式"
        logger.error(f"⚠️ 嵌入模型加载失败（{mode_label}）：{e}")
        logger.error(
            "请检查 Hugging Face 网络/代理，或预先下载模型后设置 DOCMIND_EMBEDDING_LOCAL_ONLY=1。"
        )
        raise

    logger.info(f"⚙️ BGE 向量模型运行设备: {device.upper()}")
    logger.info(f"🧩 嵌入模型加载模式: {'LOCAL_ONLY' if embedding_local_only else 'ONLINE_ALLOWED'}")
    logger.info(f"⏱️ 模型加载耗时: {time.time() - model_load_start:.2f}s")

    if not os.getenv("OPENAI_API_KEY"):
        logger.error("⚠️ [系统拦截] 未检测到大模型 API Key！请配置环境变量。")
        raise SystemExit(1)

    from google import genai

    notes_dir = _resolve_notes_dir(args, logger)
    cache_file = _resolve_cache_file(notes_dir, logger)
    change_log_file = _resolve_change_log_file(cache_file, logger)

    model_id = "gemini-2.5-flash"
    ollama_api_url = "http://localhost:11434/api/generate"
    ollama_model = "qwen2.5"

    client = genai.Client(api_key=os.getenv("OPENAI_API_KEY"), vertexai=False)

    logger.info(f"📂 当前笔记目录: {notes_dir.resolve()}")
    logger.info(f"💾 当前缓存文件: {cache_file.resolve()}")

    question_recorder = build_debug_question_recorder(notes_dir=notes_dir, logger=logger)

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
    run_chat_loop(
        repo_state,
        model_emb,
        client,
        model_id,
        ollama_api_url,
        ollama_model,
        logger,
        notes_dir=notes_dir,
        change_log_file=change_log_file,
        question_recorder=question_recorder,
    )


if __name__ == "__main__":
    main()
