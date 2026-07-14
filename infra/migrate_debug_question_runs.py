from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_RUN_FILE_PATTERN = re.compile(
    r"^questions_(?P<date>\d{8})_(?P<time>\d{6})_.*\.log$"
)


@dataclass(frozen=True)
class MigrationSummary:
    discovered: int
    flat: int
    movable: int
    moved: int
    conflicts: tuple[Path, ...]
    unsupported: tuple[Path, ...]
    by_month: dict[str, int]


def discover_run_logs(runs_dir: Path) -> tuple[Path, ...]:
    """Find both legacy flat logs and logs already stored below month folders."""
    if not runs_dir.is_dir():
        return ()
    return tuple(sorted(path for path in runs_dir.rglob("*.log") if path.is_file()))


def month_from_run_filename(filename: str) -> str | None:
    """Read the run month only from the leading timestamp, never the dataset slug."""
    match = _RUN_FILE_PATTERN.fullmatch(filename)
    if match is None:
        return None
    try:
        started_at = datetime.strptime(
            match.group("date") + match.group("time"), "%Y%m%d%H%M%S"
        )
    except ValueError:
        return None
    return started_at.strftime("%Y-%m")


def _move_without_overwrite(source: Path, destination: Path) -> None:
    """Move one file while preserving an existing destination under every code path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except FileExistsError:
        raise
    except OSError:
        destination_created = False
        try:
            with source.open("rb") as source_file:
                target_file = destination.open("xb")
                destination_created = True
                with target_file:
                    shutil.copyfileobj(source_file, target_file)
                    target_file.flush()
                    os.fsync(target_file.fileno())
            shutil.copystat(source, destination)
        except Exception:
            if destination_created:
                destination.unlink()
            raise
    source.unlink()


def migrate_flat_run_logs(debug_root: Path, *, apply: bool = False) -> MigrationSummary:
    runs_dir = debug_root / "runs"
    discovered = discover_run_logs(runs_dir)
    flat_logs = tuple(path for path in discovered if path.parent == runs_dir)
    conflicts: list[Path] = []
    unsupported: list[Path] = []
    moves: list[tuple[Path, Path, str]] = []
    by_month: Counter[str] = Counter()

    for source in flat_logs:
        month = month_from_run_filename(source.name)
        if month is None:
            unsupported.append(source)
            continue
        destination = runs_dir / month / source.name
        if destination.exists():
            conflicts.append(source)
            continue
        moves.append((source, destination, month))
        by_month[month] += 1

    moved = 0
    if apply:
        for source, destination, _month in moves:
            try:
                _move_without_overwrite(source, destination)
            except FileExistsError:
                conflicts.append(source)
                continue
            moved += 1

    return MigrationSummary(
        discovered=len(discovered),
        flat=len(flat_logs),
        movable=len(moves),
        moved=moved,
        conflicts=tuple(conflicts),
        unsupported=tuple(unsupported),
        by_month=dict(sorted(by_month.items())),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely group legacy debug-question run logs by run month."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("logs/debug_questions"),
        help="debug-question root containing runs/ (default: logs/debug_questions)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform moves; without this flag the command is a dry run",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = migrate_flat_run_logs(args.root.expanduser(), apply=args.apply)
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(
        f"[{mode}] discovered={summary.discovered} flat={summary.flat} "
        f"movable={summary.movable} moved={summary.moved}"
    )
    for month, count in summary.by_month.items():
        print(f"  {month}: {count}")
    print(
        f"  conflicts_skipped={len(summary.conflicts)} "
        f"unsupported_skipped={len(summary.unsupported)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
