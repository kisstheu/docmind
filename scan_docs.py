from pathlib import Path
from datetime import datetime

TARGET_DIR = Path(r"E:\doc")

SUPPORTED_EXT = {
    ".txt",
    ".md",
    ".docx",
    ".pdf"
}

def scan_dir(path: Path):
    for file in path.rglob("*"):
        if not file.is_file():
            continue

        if file.suffix.lower() not in SUPPORTED_EXT:
            continue

        stat = file.stat()

        size = stat.st_size
        mtime = datetime.fromtimestamp(stat.st_mtime)

        print(f"[{file.suffix}] {file.name}")
        print(f"  path: {file}")
        print(f"  size: {size}")
        print(f"  modified: {mtime}")
        print()

if __name__ == "__main__":
    scan_dir(TARGET_DIR)