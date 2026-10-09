"""SQLite database manager for assistant memory, interaction logs, and user profiles."""

import sqlite3
from pathlib import Path
from config.settings import settings
from utils.logger import logger


class DatabaseManager:
    """Manages SQLite storage for chat history, memories, and speaker interactions."""

    def __init__(self, db_path: Path = settings.database.db_path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Creates tables if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Interactions log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    speaker_name TEXT,
                    user_input TEXT,
                    assistant_response TEXT,
                    detected_emotion TEXT,
                    intent_name TEXT
                )
            """)
            # Persistent memory key-value store
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            logger.info("Database initialized successfully.")

    def log_interaction(
        self,
        speaker_name: str,
        user_input: str,
        assistant_response: str,
        detected_emotion: str = "neutral",
        intent_name: str = "general_query",
    ):
        """Records an interaction turn in the database."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO interactions 
                    (speaker_name, user_input, assistant_response, detected_emotion, intent_name)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (speaker_name, user_input, assistant_response, detected_emotion, intent_name),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log interaction to database: {e}")

    def get_memory(self, key: str) -> str | None:
        """Retrieves a stored memory by key."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM memories WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else None

    def set_memory(self, key: str, value: str):
        """Saves or updates a memory value."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memories (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
                """,
                (key, value),
            )
            conn.commit()
