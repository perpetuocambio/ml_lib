"""
Base database manager for SQLite operations.

Provides reusable database connection management and transaction handling.
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BaseDatabaseManager:
    """
    Base class for SQLite database managers.

    Provides common database operations and connection management.
    """

    def __init__(self, db_path: Path | str, auto_init: bool = True):
        """
        Initialize database manager.

        Args:
            db_path: Path to SQLite database file
            auto_init: Automatically initialize schema
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if auto_init:
            self._init_schema()

        logger.debug(f"{self.__class__.__name__} initialized: {self.db_path}")

    def _init_schema(self):
        """Initialize database schema. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement _init_schema()")

    @contextmanager
    def connection(self):
        """
        Get database connection with automatic commit/rollback.

        Yields:
            sqlite3.Connection with row_factory enabled

        Example:
            >>> with db.connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM models")
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def execute(self, query: str, params: tuple = ()) -> list[sqlite3.Row]:
        """
        Execute a query and return all results.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of Row objects

        Example:
            >>> rows = db.execute("SELECT * FROM models WHERE type = ?", ("lora",))
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()

    def execute_one(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """
        Execute a query and return first result.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Row object or None

        Example:
            >>> row = db.execute_one("SELECT * FROM models WHERE id = ?", (model_id,))
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()

    def execute_many(self, query: str, params_list: list[tuple]) -> int:
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query string
            params_list: List of parameter tuples

        Returns:
            Number of rows affected

        Example:
            >>> count = db.execute_many(
            ...     "INSERT INTO tags (tag) VALUES (?)",
            ...     [("tag1",), ("tag2",), ("tag3",)]
            ... )
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            return cursor.rowcount

    def insert(self, table: str, data: dict[str, Any]) -> Optional[int]:
        """
        Insert a row into a table.

        Args:
            table: Table name
            data: Dictionary of column: value

        Returns:
            Last insert rowid or None

        Example:
            >>> row_id = db.insert("models", {
            ...     "model_id": "123",
            ...     "name": "My Model",
            ...     "type": "lora"
            ... })
        """
        columns = ", ".join(data.keys())
        placeholders = ", ".join("?" * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, tuple(data.values()))
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Insert failed: {e}")
            return None

    def update(self, table: str, data: dict[str, Any], where: str, where_params: tuple) -> int:
        """
        Update rows in a table.

        Args:
            table: Table name
            data: Dictionary of column: value to update
            where: WHERE clause (without "WHERE")
            where_params: Parameters for WHERE clause

        Returns:
            Number of rows updated

        Example:
            >>> count = db.update(
            ...     "models",
            ...     {"rating": 4.5},
            ...     "model_id = ?",
            ...     ("123",)
            ... )
        """
        set_clause = ", ".join(f"{k} = ?" for k in data.keys())
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"
        params = tuple(data.values()) + where_params

        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Update failed: {e}")
            return 0

    def delete(self, table: str, where: str, where_params: tuple) -> int:
        """
        Delete rows from a table.

        Args:
            table: Table name
            where: WHERE clause (without "WHERE")
            where_params: Parameters for WHERE clause

        Returns:
            Number of rows deleted

        Example:
            >>> count = db.delete("models", "model_id = ?", ("123",))
        """
        query = f"DELETE FROM {table} WHERE {where}"

        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, where_params)
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return 0

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        row = self.execute_one(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        return row is not None

    def get_table_info(self, table_name: str) -> list[dict]:
        """Get column information for a table."""
        rows = self.execute(f"PRAGMA table_info({table_name})")
        return [dict(row) for row in rows]

    def vacuum(self):
        """Optimize database file size."""
        with self.connection() as conn:
            conn.execute("VACUUM")
            logger.info("Database vacuumed")
