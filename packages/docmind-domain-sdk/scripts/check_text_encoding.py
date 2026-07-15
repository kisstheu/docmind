from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path


TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".json", ".toml"}
ALLOWED_CONTROLS = {"\t", "\n", "\r"}


def iter_text_files(inputs: list[Path]):
    for path in inputs:
        if path.is_dir():
            yield from (
                child
                for child in sorted(path.rglob("*"))
                if child.is_file() and child.suffix.lower() in TEXT_SUFFIXES
            )
        elif path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def validate_file(path: Path) -> list[str]:
    try:
        text = path.read_bytes().decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return [f"{path}: invalid UTF-8: {exc}"]
    errors: list[str] = []
    if "\ufffd" in text:
        errors.append(f"{path}: contains replacement character U+FFFD")
    for index, character in enumerate(text):
        if unicodedata.category(character) == "Cc" and character not in ALLOWED_CONTROLS:
            errors.append(
                f"{path}: unexpected control character U+{ord(character):04X} at offset {index}"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    files = list(dict.fromkeys(iter_text_files(args.paths)))
    errors = [error for path in files for error in validate_file(path)]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"text encoding check: ok ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
