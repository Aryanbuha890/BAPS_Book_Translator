import os
import sqlite3
import hashlib

class ProgressDB:
    """
    SQLite Database helper for managing book translation progress,
    manual edits, and user glossaries.
    """
    def __init__(self, book_path: str):
        # Create a clean database name from the book file path
        base_name = os.path.basename(book_path)
        safe_name = "".join([c if c.isalnum() else "_" for c in base_name])
        # Ensure we have a db directory
        db_dir = os.path.join(os.path.dirname(os.path.abspath(book_path)), "db")
        os.makedirs(db_dir, exist_ok=True)
        
        self.db_path = os.path.join(db_dir, f"{safe_name}_progress.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        """
        Creates necessary tables for progress tracking and glossary.
        """
        with self.conn:
            # Table for tracking translation of each sentence chunk
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS translation_progress (
                    chunk_id INTEGER PRIMARY KEY,
                    chapter_index INTEGER,
                    chapter_title TEXT,
                    original_text TEXT,
                    translated_text TEXT,
                    status TEXT DEFAULT 'pending', -- 'pending', 'translating', 'done'
                    modified_by_user INTEGER DEFAULT 0 -- 1 if manually overwritten
                )
            """)
            # Table for custom glossary substitutions
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS glossary (
                    term_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_term TEXT UNIQUE,
                    translated_term TEXT
                )
            """)

    def initialize_chunks(self, chapters: list[tuple[str, list[str]]]) -> bool:
        """
        Populates the DB with text chunks in order if it is not already initialized.
        Returns True if newly initialized, False if progress already exists.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM translation_progress")
        count = cursor.fetchone()[0]
        
        if count > 0:
            # DB is already populated, resume mode
            return False
            
        # DB is empty, let's insert all chunks
        chunk_id = 0
        with self.conn:
            for ch_idx, (ch_title, chunks) in enumerate(chapters):
                for chunk in chunks:
                    self.conn.execute("""
                        INSERT INTO translation_progress 
                        (chunk_id, chapter_index, chapter_title, original_text, translated_text, status, modified_by_user)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (chunk_id, ch_idx, ch_title, chunk, "", "pending", 0))
                    chunk_id += 1
        return True

    def get_pending_chunks(self) -> list[tuple[int, str]]:
        """
        Retrieves all chunks that still need to be translated.
        Returns a list of tuples: [(chunk_id, original_text), ...]
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT chunk_id, original_text 
            FROM translation_progress 
            WHERE status != 'done' 
            ORDER BY chunk_id ASC
        """)
        return cursor.fetchall()

    def update_chunk_translating(self, chunk_id: int):
        """
        Marks a chunk as currently undergoing translation to prevent duplicates.
        """
        with self.conn:
            self.conn.execute("""
                UPDATE translation_progress 
                SET status = 'translating' 
                WHERE chunk_id = ? AND status != 'done'
            """, (chunk_id,))

    def save_chunk(self, chunk_id: int, translated_text: str):
        """
        Saves translation result for a chunk and marks it as done.
        """
        with self.conn:
            self.conn.execute("""
                UPDATE translation_progress 
                SET translated_text = ?, status = 'done' 
                WHERE chunk_id = ?
            """, (translated_text, chunk_id))

    def mark_chunk_done(self, chunk_id: int, translated_text: str, modified_by_user: int = 0):
        """
        Explicitly saves translation, sets state to 'done', and marks if modified by user.
        """
        with self.conn:
            self.conn.execute("""
                UPDATE translation_progress 
                SET translated_text = ?, status = 'done', modified_by_user = ? 
                WHERE chunk_id = ?
            """, (translated_text, modified_by_user, chunk_id))

    def get_all_chunks(self) -> list[dict]:
        """
        Retrieves all database rows ordered by chunk_id.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT chunk_id, chapter_index, chapter_title, original_text, translated_text, status, modified_by_user 
            FROM translation_progress 
            ORDER BY chunk_id ASC
        """)
        columns = [col[0] for col in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def get_assembled_chapters(self) -> list[tuple[str, list[str]]]:
        """
        Retrieves fully-translated structured text grouped by chapter.
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT chapter_index, chapter_title, translated_text 
            FROM translation_progress 
            ORDER BY chunk_id ASC
        """)
        rows = cursor.fetchall()
        
        chapters_dict = {}
        for ch_idx, ch_title, text in rows:
            if ch_idx not in chapters_dict:
                chapters_dict[ch_idx] = (ch_title, [])
            chapters_dict[ch_idx][1].append(text)
            
        sorted_indices = sorted(chapters_dict.keys())
        return [chapters_dict[idx] for idx in sorted_indices]

    def is_complete(self) -> bool:
        """
        Checks if all chunks are fully translated.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM translation_progress WHERE status != 'done'")
        pending_count = cursor.fetchone()[0]
        return pending_count == 0

    def reset_progress(self, force_all: bool = False):
        """
        Resets progress. If force_all is False, it will NOT reset chunks modified by the user.
        """
        with self.conn:
            if force_all:
                self.conn.execute("UPDATE translation_progress SET translated_text = '', status = 'pending', modified_by_user = 0")
            else:
                self.conn.execute("""
                    UPDATE translation_progress 
                    SET translated_text = '', status = 'pending' 
                    WHERE modified_by_user = 0
                """)

    def get_progress_stats(self) -> dict:
        """
        Calculates and returns progress statistics.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM translation_progress")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM translation_progress WHERE status = 'done'")
        completed = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM translation_progress WHERE modified_by_user = 1")
        user_edits = cursor.fetchone()[0]
        
        percent = (completed / total * 100.0) if total > 0 else 0.0
        
        return {
            "total_chunks": total,
            "completed_chunks": completed,
            "pending_chunks": total - completed,
            "percent_complete": percent,
            "user_edited_chunks": user_edits
        }

    # --- Glossary Methods ---
    def save_glossary_term(self, original_term: str, translated_term: str):
        """
        Adds or updates a glossary word mapping.
        """
        with self.conn:
            self.conn.execute("""
                INSERT INTO glossary (original_term, translated_term)
                VALUES (?, ?)
                ON CONFLICT(original_term) DO UPDATE SET translated_term = excluded.translated_term
            """, (original_term.strip(), translated_term.strip()))

    def delete_glossary_term(self, original_term: str):
        """
        Deletes a glossary word mapping.
        """
        with self.conn:
            self.conn.execute("DELETE FROM glossary WHERE original_term = ?", (original_term.strip(),))

    def get_glossary(self) -> dict[str, str]:
        """
        Retrieves all glossary mappings.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT original_term, translated_term FROM glossary")
        return {row[0]: row[1] for row in cursor.fetchall()}

    def close(self):
        """
        Closes connection.
        """
        self.conn.close()
