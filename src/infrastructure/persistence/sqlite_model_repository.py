"""SQLite Model Repository - Production persistence implementation.

This implementation provides real persistence with SQLite database,
suitable for production use.
"""

import sqlite3
from typing import Optional
from pathlib import Path
from contextlib import contextmanager

from ml_lib.diffusion.domain.repositories.model_repository import IModelRepository
from ml_lib.diffusion.domain.entities.lora import LoRA


class SQLiteModelRepository(IModelRepository):
    """
    SQLite implementation of IModelRepository.

    Provides full CRUD operations with real database persistence.
    Thread-safe with connection-per-operation pattern.
    """

    def __init__(self, db_path: Path | str):
        """
        Initialize repository.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema if not exists."""
        schema_path = Path(__file__).parent / "schema.sql"

        with self._get_connection() as conn:
            with open(schema_path, "r") as f:
                schema = f.read()
            conn.executescript(schema)
            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection (context manager for auto-close)."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # Access columns by name
        try:
            yield conn
        finally:
            conn.close()

    def get_lora_by_name(self, name: str) -> Optional[LoRA]:
        """Get LoRA by name."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM loras WHERE name = ?",
                (name,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_lora(row, conn)

    def get_all_loras(self) -> list[LoRA]:
        """Get all LoRAs."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM loras")
            rows = cursor.fetchall()

            return [self._row_to_lora(row, conn) for row in rows]

    def get_loras_by_base_model(self, base_model: str) -> list[LoRA]:
        """Get LoRAs compatible with base model."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM loras WHERE base_model = ?",
                (base_model,)
            )
            rows = cursor.fetchall()

            return [self._row_to_lora(row, conn) for row in rows]

    def get_loras_by_tags(self, tags: list[str]) -> list[LoRA]:
        """Get LoRAs that have any of the specified tags."""
        if not tags:
            return []

        tags_lower = [t.lower() for t in tags]
        placeholders = ",".join("?" * len(tags_lower))

        with self._get_connection() as conn:
            query = f"""
                SELECT DISTINCT l.*
                FROM loras l
                JOIN lora_tags lt ON l.id = lt.lora_id
                WHERE LOWER(lt.tag) IN ({placeholders})
            """
            cursor = conn.execute(query, tags_lower)
            rows = cursor.fetchall()

            return [self._row_to_lora(row, conn) for row in rows]

    def get_popular_loras(self, limit: int = 10) -> list[LoRA]:
        """Get most popular LoRAs by download count."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM loras
                ORDER BY download_count DESC, rating DESC
                LIMIT ?
                """,
                (limit,)
            )
            rows = cursor.fetchall()

            return [self._row_to_lora(row, conn) for row in rows]

    def search_loras(
        self,
        query: str,
        base_model: Optional[str] = None,
        min_rating: float = 0.0,
        limit: int = 20,
    ) -> list[LoRA]:
        """Search LoRAs by query."""
        query_lower = query.lower()

        with self._get_connection() as conn:
            # Build dynamic query
            sql = """
                SELECT DISTINCT l.*
                FROM loras l
                LEFT JOIN lora_tags lt ON l.id = lt.lora_id
                LEFT JOIN lora_trigger_words ltw ON l.id = ltw.lora_id
                WHERE (
                    LOWER(l.name) LIKE ? OR
                    LOWER(lt.tag) LIKE ? OR
                    LOWER(ltw.trigger_word) LIKE ?
                )
                AND l.rating >= ?
            """
            params = [f"%{query_lower}%"] * 3 + [min_rating]

            if base_model:
                sql += " AND l.base_model = ?"
                params.append(base_model)

            sql += " ORDER BY l.rating DESC, l.download_count DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()

            return [self._row_to_lora(row, conn) for row in rows]

    def add_lora(self, lora: LoRA) -> None:
        """Add a new LoRA to repository."""
        with self._get_connection() as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT id FROM loras WHERE name = ?",
                (lora.name,)
            ).fetchone()

            if existing:
                raise ValueError(f"LoRA '{lora.name}' already exists")

            # Insert main record
            cursor = conn.execute(
                """
                INSERT INTO loras (name, path, base_model, weight, download_count, rating)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    lora.name,
                    str(lora.path),
                    lora.base_model,
                    lora.weight.value,
                    lora.download_count,
                    lora.rating,
                )
            )
            lora_id = cursor.lastrowid

            # Insert trigger words
            for trigger in lora.trigger_words:
                conn.execute(
                    "INSERT INTO lora_trigger_words (lora_id, trigger_word) VALUES (?, ?)",
                    (lora_id, trigger)
                )

            # Insert tags
            for tag in lora.tags:
                conn.execute(
                    "INSERT INTO lora_tags (lora_id, tag) VALUES (?, ?)",
                    (lora_id, tag)
                )

            conn.commit()

    def update_lora(self, lora: LoRA) -> None:
        """Update existing LoRA."""
        with self._get_connection() as conn:
            # Get existing ID
            row = conn.execute(
                "SELECT id FROM loras WHERE name = ?",
                (lora.name,)
            ).fetchone()

            if row is None:
                raise ValueError(f"LoRA '{lora.name}' not found")

            lora_id = row["id"]

            # Update main record
            conn.execute(
                """
                UPDATE loras
                SET path = ?, base_model = ?, weight = ?, download_count = ?, rating = ?
                WHERE id = ?
                """,
                (
                    str(lora.path),
                    lora.base_model,
                    lora.weight.value,
                    lora.download_count,
                    lora.rating,
                    lora_id,
                )
            )

            # Delete old trigger words and tags
            conn.execute("DELETE FROM lora_trigger_words WHERE lora_id = ?", (lora_id,))
            conn.execute("DELETE FROM lora_tags WHERE lora_id = ?", (lora_id,))

            # Insert new trigger words
            for trigger in lora.trigger_words:
                conn.execute(
                    "INSERT INTO lora_trigger_words (lora_id, trigger_word) VALUES (?, ?)",
                    (lora_id, trigger)
                )

            # Insert new tags
            for tag in lora.tags:
                conn.execute(
                    "INSERT INTO lora_tags (lora_id, tag) VALUES (?, ?)",
                    (lora_id, tag)
                )

            conn.commit()

    def delete_lora(self, name: str) -> bool:
        """Delete LoRA by name."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM loras WHERE name = ?",
                (name,)
            )
            conn.commit()

            return cursor.rowcount > 0

    def count_loras(self) -> int:
        """Get total count of LoRAs."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) as count FROM loras")
            row = cursor.fetchone()
            return row["count"]

    def _row_to_lora(self, row: sqlite3.Row, conn: sqlite3.Connection) -> LoRA:
        """
        Convert database row to LoRA entity.

        Args:
            row: Database row
            conn: Database connection (for fetching related data)

        Returns:
            LoRA entity
        """
        lora_id = row["id"]

        # Fetch trigger words
        trigger_cursor = conn.execute(
            "SELECT trigger_word FROM lora_trigger_words WHERE lora_id = ?",
            (lora_id,)
        )
        trigger_words = [r["trigger_word"] for r in trigger_cursor.fetchall()]

        # Fetch tags
        tags_cursor = conn.execute(
            "SELECT tag FROM lora_tags WHERE lora_id = ?",
            (lora_id,)
        )
        tags = [r["tag"] for r in tags_cursor.fetchall()]

        # Create LoRA entity
        return LoRA.create(
            name=row["name"],
            path=Path(row["path"]),
            base_model=row["base_model"],
            weight=row["weight"],
            trigger_words=trigger_words,
            tags=tags,
            download_count=row["download_count"],
            rating=row["rating"],
        )
