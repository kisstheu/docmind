from __future__ import annotations

import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def build_logger(
    name: str = "DocMind",
    log_file: str = "logs/docmind_sys.log",
    console_level: int = logging.DEBUG,
    file_level: int = logging.DEBUG,
    backup_count: int = 8,
    rotate_when: str = "midnight",
    rotate_interval: int = 1,
    rotate_utc: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    log_file = os.getenv("DOCMIND_LOG_FILE", log_file)
    backup_count = int(os.getenv("DOCMIND_LOG_BACKUP_COUNT", str(backup_count)))
    rotate_when = os.getenv("DOCMIND_LOG_ROTATE_WHEN", rotate_when)
    rotate_interval = int(os.getenv("DOCMIND_LOG_ROTATE_INTERVAL", str(rotate_interval)))
    rotate_utc = os.getenv("DOCMIND_LOG_ROTATE_UTC", str(rotate_utc)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

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

    file_handler = TimedRotatingFileHandler(
        log_path,
        when=rotate_when,
        interval=rotate_interval,
        backupCount=backup_count,
        encoding="utf-8",
        utc=rotate_utc,
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(file_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
    )

    logger.addHandler(file_handler)

    return logger
