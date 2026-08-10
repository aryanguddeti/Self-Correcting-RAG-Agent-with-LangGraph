import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from src.core.config import settings

DB_PATH = settings.PROJECT_ROOT / "data" / "ingestion_ledger.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS ingested_files (
    file_name         TEXT PRIMARY KEY,
    content_hash      TEXT NOT NULL,
    status            TEXT NOT NULL,
    chunk_count       INTEGER NOT NULL DEFAULT 0,
    last_indexed_at   TIMESTAMP NOT NULL
)
"""


class LedgerManager:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(CREATE_TABLE_SQL)

    def get_file_record(self, file_name: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM ingested_files WHERE file_name = ?", (file_name,)
            ).fetchone()
        return dict(row) if row else None

    def upsert_file_record(
        self, file_name: str, content_hash: str, chunk_count: int, status: str
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingested_files (file_name, content_hash, status, chunk_count, last_indexed_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(file_name) DO UPDATE SET
                    content_hash    = excluded.content_hash,
                    status          = excluded.status,
                    chunk_count     = excluded.chunk_count,
                    last_indexed_at = excluded.last_indexed_at
                """,
                (file_name, content_hash, status, chunk_count, now),
            )

    def delete_file_record(self, file_name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM ingested_files WHERE file_name = ?", (file_name,)
            )

    def get_all_file_names(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT file_name FROM ingested_files").fetchall()
        return [row["file_name"] for row in rows]
