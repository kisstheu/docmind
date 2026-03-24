from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def build_logger(
    name: str = "DocMind",
    log_file: str = "logs/docmind_sys.log",
    console_level: int = logging.DEBUG,
    file_level: int = logging.DEBUG,
    max_bytes: int = 256 * 1024,
    backup_count: int = 8,
) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_file = os.getenv("DOCMIND_LOG_FILE", log_file)
    max_bytes = int(os.getenv("DOCMIND_LOG_MAX_BYTES", str(max_bytes)))
    backup_count = int(os.getenv("DOCMIND_LOG_BACKUP_COUNT", str(backup_count)))

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(
        logging.Formatter("%(message)s")
    )
    logger.addHandler(console_handler)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
    )

    if log_path.exists() and log_path.stat().st_size >= max_bytes:
        file_handler.doRollover()

    logger.addHandler(file_handler)

    return logger
