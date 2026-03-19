from __future__ import annotations

import os


def apply_environment_defaults() -> None:
    os.environ.setdefault("http_proxy", "http://127.0.0.1:7897")
    os.environ.setdefault("https_proxy", "http://127.0.0.1:7897")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
