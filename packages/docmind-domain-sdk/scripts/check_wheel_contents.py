from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


REQUIRED = {
    "docmind_domain_sdk/py.typed",
    "docmind_domain_sdk/schemas/protocol-1.0.schema.json",
}
FORBIDDEN_PARTS = {"tests", "spikes", "scripts", "__pycache__"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.wheel) as archive:
        names = set(archive.namelist())
    missing = sorted(REQUIRED - names)
    forbidden = sorted(
        name
        for name in names
        if FORBIDDEN_PARTS.intersection(Path(name).parts)
        or name.endswith((".pyc", ".pyo"))
    )
    if missing or forbidden:
        if missing:
            print(f"missing wheel entries: {missing}")
        if forbidden:
            print(f"forbidden wheel entries: {forbidden}")
        return 1
    print(f"wheel content check: ok ({args.wheel})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
