from __future__ import annotations

import datetime
import hashlib
import sqlite3
import uuid
from pathlib import Path


def collect_file_snapshot(abs_path: Path, notes_dir: Path) -> dict:
    stat = abs_path.stat()
    rel_path = abs_path.relative_to(notes_dir).as_posix()

    sha256 = hashlib.sha256()
    with abs_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)

    return {
        "relative_path": rel_path,
        "size": int(stat.st_size),
        "mtime": float(stat.st_mtime),
        "ctime": float(stat.st_ctime),
        "sha256": sha256.hexdigest(),
    }


class FileChangeStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS files (
                    file_id TEXT PRIMARY KEY,
                    notes_dir TEXT NOT NULL,
                    original_path TEXT NOT NULL,
                    current_path TEXT NOT NULL,
                    original_size INTEGER NOT NULL,
                    original_mtime REAL NOT NULL,
                    original_ctime REAL NOT NULL,
                    original_sha256 TEXT NOT NULL,
                    last_size INTEGER NOT NULL,
                    last_mtime REAL NOT NULL,
                    last_ctime REAL NOT NULL,
                    last_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS file_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    notes_dir TEXT NOT NULL,
                    before_path TEXT NOT NULL,
                    after_path TEXT NOT NULL,
                    before_size INTEGER NOT NULL,
                    before_mtime REAL NOT NULL,
                    before_ctime REAL NOT NULL,
                    before_sha256 TEXT NOT NULL,
                    after_size INTEGER NOT NULL,
                    after_mtime REAL NOT NULL,
                    after_ctime REAL NOT NULL,
                    after_sha256 TEXT NOT NULL,
                    reason TEXT,
                    requested_text TEXT,
                    confirmed_text TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_current ON files(notes_dir, current_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_file ON file_events(file_id, created_at)")
            conn.commit()

    def record_rename(
        self,
        *,
        notes_dir: Path,
        before: dict,
        after: dict,
        reason: str = "",
        requested_text: str = "",
        confirmed_text: str = "",
    ) -> int:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        notes_dir_key = str(notes_dir.resolve())
        before_path = before["relative_path"]

        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_id FROM files WHERE notes_dir = ? AND current_path = ?",
                (notes_dir_key, before_path),
            ).fetchone()

            if row:
                file_id = row[0]
                conn.execute(
                    """
                    UPDATE files
                    SET current_path = ?,
                        last_size = ?,
                        last_mtime = ?,
                        last_ctime = ?,
                        last_sha256 = ?,
                        updated_at = ?
                    WHERE file_id = ?
                    """,
                    (
                        after["relative_path"],
                        int(after["size"]),
                        float(after["mtime"]),
                        float(after["ctime"]),
                        after["sha256"],
                        now,
                        file_id,
                    ),
                )
            else:
                file_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO files (
                        file_id, notes_dir, original_path, current_path,
                        original_size, original_mtime, original_ctime, original_sha256,
                        last_size, last_mtime, last_ctime, last_sha256,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        notes_dir_key,
                        before["relative_path"],
                        after["relative_path"],
                        int(before["size"]),
                        float(before["mtime"]),
                        float(before["ctime"]),
                        before["sha256"],
                        int(after["size"]),
                        float(after["mtime"]),
                        float(after["ctime"]),
                        after["sha256"],
                        now,
                        now,
                    ),
                )

            cur = conn.execute(
                """
                INSERT INTO file_events (
                    file_id, event_type, notes_dir,
                    before_path, after_path,
                    before_size, before_mtime, before_ctime, before_sha256,
                    after_size, after_mtime, after_ctime, after_sha256,
                    reason, requested_text, confirmed_text, created_at
                )
                VALUES (?, 'rename', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    notes_dir_key,
                    before["relative_path"],
                    after["relative_path"],
                    int(before["size"]),
                    float(before["mtime"]),
                    float(before["ctime"]),
                    before["sha256"],
                    int(after["size"]),
                    float(after["mtime"]),
                    float(after["ctime"]),
                    after["sha256"],
                    reason,
                    requested_text,
                    confirmed_text,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def record_delete(
        self,
        *,
        notes_dir: Path,
        before: dict,
        after: dict,
        reason: str = "",
        requested_text: str = "",
        confirmed_text: str = "",
    ) -> int:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        notes_dir_key = str(notes_dir.resolve())
        before_path = before["relative_path"]

        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_id FROM files WHERE notes_dir = ? AND current_path = ?",
                (notes_dir_key, before_path),
            ).fetchone()

            if row:
                file_id = row[0]
                conn.execute(
                    """
                    UPDATE files
                    SET current_path = ?,
                        last_size = ?,
                        last_mtime = ?,
                        last_ctime = ?,
                        last_sha256 = ?,
                        updated_at = ?
                    WHERE file_id = ?
                    """,
                    (
                        after["relative_path"],
                        int(after["size"]),
                        float(after["mtime"]),
                        float(after["ctime"]),
                        after["sha256"],
                        now,
                        file_id,
                    ),
                )
            else:
                file_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO files (
                        file_id, notes_dir, original_path, current_path,
                        original_size, original_mtime, original_ctime, original_sha256,
                        last_size, last_mtime, last_ctime, last_sha256,
                        created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        notes_dir_key,
                        before["relative_path"],
                        after["relative_path"],
                        int(before["size"]),
                        float(before["mtime"]),
                        float(before["ctime"]),
                        before["sha256"],
                        int(after["size"]),
                        float(after["mtime"]),
                        float(after["ctime"]),
                        after["sha256"],
                        now,
                        now,
                    ),
                )

            cur = conn.execute(
                """
                INSERT INTO file_events (
                    file_id, event_type, notes_dir,
                    before_path, after_path,
                    before_size, before_mtime, before_ctime, before_sha256,
                    after_size, after_mtime, after_ctime, after_sha256,
                    reason, requested_text, confirmed_text, created_at
                )
                VALUES (?, 'delete', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    notes_dir_key,
                    before["relative_path"],
                    after["relative_path"],
                    int(before["size"]),
                    float(before["mtime"]),
                    float(before["ctime"]),
                    before["sha256"],
                    int(after["size"]),
                    float(after["mtime"]),
                    float(after["ctime"]),
                    after["sha256"],
                    reason,
                    requested_text,
                    confirmed_text,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_recent_renames(self, *, notes_dir: Path, limit: int = 20) -> list[dict]:
        notes_dir_key = str(notes_dir.resolve())
        safe_limit = max(1, min(int(limit), 200))

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    event_id,
                    before_path,
                    after_path,
                    before_sha256,
                    after_sha256,
                    created_at
                FROM file_events
                WHERE notes_dir = ? AND event_type = 'rename'
                ORDER BY event_id DESC
                LIMIT ?
                """,
                (notes_dir_key, safe_limit),
            ).fetchall()

        return [
            {
                "event_id": int(row[0]),
                "before_path": row[1],
                "after_path": row[2],
                "before_sha256": row[3],
                "after_sha256": row[4],
                "created_at": row[5],
            }
            for row in rows
        ]

    def list_recent_events(self, *, notes_dir: Path, limit: int = 20) -> list[dict]:
        notes_dir_key = str(notes_dir.resolve())
        safe_limit = max(1, min(int(limit), 500))

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    event_id,
                    event_type,
                    before_path,
                    after_path,
                    before_sha256,
                    after_sha256,
                    reason,
                    created_at
                FROM file_events
                WHERE notes_dir = ?
                ORDER BY event_id DESC
                LIMIT ?
                """,
                (notes_dir_key, safe_limit),
            ).fetchall()

        return [
            {
                "event_id": int(row[0]),
                "event_type": row[1],
                "before_path": row[2],
                "after_path": row[3],
                "before_sha256": row[4],
                "after_sha256": row[5],
                "reason": row[6] or "",
                "created_at": row[7],
            }
            for row in rows
        ]
