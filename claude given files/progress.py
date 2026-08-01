"""
progress.py

SQLite-backed progress tracking so large books can be interrupted and
resumed without re-translating already-completed chunks.
"""

import sqlite3
from pathlib import Path


class ProgressDB:
    def __init__(self, book_path: str):
        db_name = Path(book_path).stem + "_progress.db"
        self.db_path = Path(book_path).parent / db_name
        self._conn = sqlite3.connect(self.db_path)
        self._init_schema()

    def _init_schema(self):
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY,
                chapter_index INTEGER NOT NULL,
                original_text TEXT NOT NULL,
                translated_text TEXT,
                status TEXT NOT NULL DEFAULT 'pending'
            )
            """
        )
        self._conn.commit()

    def seed_if_empty(self, chapter_chunks: list[tuple[int, str]]):
        """
        Populate the DB with all chunks on first run only.
        chapter_chunks: list of (chapter_index, original_text) in order.
        No-op if the DB already has rows (i.e. resuming).
        """
        existing = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if existing > 0:
            return
        self._conn.executemany(
            "INSERT INTO chunks (chapter_index, original_text, status) VALUES (?, ?, 'pending')",
            chapter_chunks,
        )
        self._conn.commit()

    def save_chunk(self, chunk_id: int, translated_text: str):
        self._conn.execute(
            "UPDATE chunks SET translated_text = ?, status = 'done' WHERE chunk_id = ?",
            (translated_text, chunk_id),
        )
        self._conn.commit()

    def get_pending_chunks(self) -> list[tuple[int, str]]:
        rows = self._conn.execute(
            "SELECT chunk_id, original_text FROM chunks WHERE status = 'pending' ORDER BY chunk_id"
        ).fetchall()
        return rows

    def get_all_chunks(self) -> list[tuple[int, int, str, str, str]]:
        """Returns (chunk_id, chapter_index, original_text, translated_text, status), ordered."""
        return self._conn.execute(
            "SELECT chunk_id, chapter_index, original_text, translated_text, status "
            "FROM chunks ORDER BY chunk_id"
        ).fetchall()

    def is_complete(self) -> bool:
        pending = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE status = 'pending'"
        ).fetchone()[0]
        return pending == 0

    def total_and_done(self) -> tuple[int, int]:
        total = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        done = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE status = 'done'"
        ).fetchone()[0]
        return total, done

    def close(self):
        self._conn.close()
