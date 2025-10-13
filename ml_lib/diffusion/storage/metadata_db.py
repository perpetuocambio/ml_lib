"""
SQLite database for model metadata storage.

Replaces dependency on ComfyUI's metadata.json files with a centralized database.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import enums from separate file to avoid circular import
from ml_lib.diffusion.model_enums import ModelType, BaseModel, Source, ModelFormat
from ml_lib.storage.base_db import BaseDatabaseManager

logger = logging.getLogger(__name__)


class MetadataDatabase(BaseDatabaseManager):
    """
    SQLite database for storing and querying model metadata.

    Provides independence from ComfyUI's metadata.json format.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: Path | str = None):
        """
        Initialize metadata database.

        Args:
            db_path: Path to SQLite database file (default: ml_lib/data/models.db)
        """
        if db_path is None:
            # Default to data directory
            db_path = Path(__file__).parent.parent.parent / "data" / "models.db"

        # Call parent constructor
        super().__init__(db_path, auto_init=True)

        logger.info(f"MetadataDatabase initialized: {self.db_path}")

    def _init_schema(self):
        """Initialize database schema."""
        with self.connection() as conn:
            cursor = conn.cursor()

            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source TEXT NOT NULL,
                    type TEXT NOT NULL,
                    base_model TEXT NOT NULL,
                    version TEXT DEFAULT 'main',

                    -- Technical
                    format TEXT DEFAULT 'safetensors',
                    size_bytes INTEGER DEFAULT 0,
                    sha256 TEXT,

                    -- Semantic
                    description TEXT,

                    -- Stats
                    download_count INTEGER DEFAULT 0,
                    rating REAL DEFAULT 0.0,
                    recommended_weight REAL,

                    -- Paths
                    local_path TEXT,
                    remote_url TEXT,

                    -- Timestamps
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                    -- Indexes
                    UNIQUE(model_id)
                )
            """)

            # Tags table (many-to-many)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag TEXT UNIQUE NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_tags (
                    model_id TEXT NOT NULL,
                    tag_id INTEGER NOT NULL,
                    PRIMARY KEY (model_id, tag_id),
                    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE,
                    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
                )
            """)

            # Trigger words table (many-to-many)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trigger_words (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    word TEXT UNIQUE NOT NULL
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_trigger_words (
                    model_id TEXT NOT NULL,
                    word_id INTEGER NOT NULL,
                    PRIMARY KEY (model_id, word_id),
                    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE,
                    FOREIGN KEY (word_id) REFERENCES trigger_words(id) ON DELETE CASCADE
                )
            """)

            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_type ON models(type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_base_model ON models(base_model)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_source ON models(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_rating ON models(rating DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_downloads ON models(download_count DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_local_path ON models(local_path)")

            # Schema version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_info (
                    version INTEGER PRIMARY KEY,
                    migrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # User overrides table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_overrides (
                    model_id TEXT PRIMARY KEY,
                    override_data TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE
                )
            """)

            cursor.execute("INSERT OR IGNORE INTO schema_info (version) VALUES (?)", (self.SCHEMA_VERSION,))

            conn.commit()
            logger.info("Database schema initialized")

    def insert_model(self, metadata) -> bool:
        """
        Insert or update model metadata.

        Args:
            metadata: ModelMetadata to store

        Returns:
            True if successful
        """
        try:
            with self.connection() as conn:
                cursor = conn.cursor()

                # Insert/update model
                cursor.execute("""
                    INSERT OR REPLACE INTO models (
                        model_id, name, source, type, base_model, version,
                        format, size_bytes, sha256, description,
                        download_count, rating, recommended_weight,
                        local_path, remote_url,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.model_id,
                    metadata.name,
                    metadata.source.value,
                    metadata.type.value,
                    metadata.base_model.value,
                    metadata.version,
                    metadata.format.value,
                    metadata.size_bytes,
                    metadata.sha256,
                    metadata.description,
                    metadata.download_count,
                    metadata.rating,
                    metadata.recommended_weight,
                    str(metadata.local_path) if metadata.local_path else None,
                    metadata.remote_url,
                    metadata.created_at.isoformat(),
                    metadata.updated_at.isoformat(),
                ))

                # Insert tags
                for tag in metadata.tags:
                    cursor.execute("INSERT OR IGNORE INTO tags (tag) VALUES (?)", (tag,))
                    cursor.execute("SELECT id FROM tags WHERE tag = ?", (tag,))
                    tag_id = cursor.fetchone()[0]
                    cursor.execute(
                        "INSERT OR IGNORE INTO model_tags (model_id, tag_id) VALUES (?, ?)",
                        (metadata.model_id, tag_id)
                    )

                # Insert trigger words
                for word in metadata.trigger_words:
                    cursor.execute("INSERT OR IGNORE INTO trigger_words (word) VALUES (?)", (word,))
                    cursor.execute("SELECT id FROM trigger_words WHERE word = ?", (word,))
                    word_id = cursor.fetchone()[0]
                    cursor.execute(
                        "INSERT OR IGNORE INTO model_trigger_words (model_id, word_id) VALUES (?, ?)",
                        (metadata.model_id, word_id)
                    )

                conn.commit()
                return True

        except Exception as e:
            logger.error(f"Failed to insert model {metadata.model_id}: {e}")
            return False

    def get_model(self, model_id: str):
        """
        Get model metadata by ID.

        Args:
            model_id: Model identifier

        Returns:
            ModelMetadata or None if not found
        """
        try:
            with self.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM models WHERE model_id = ?", (model_id,))
                row = cursor.fetchone()

                if not row:
                    return None

                # Get tags
                cursor.execute("""
                    SELECT t.tag FROM tags t
                    JOIN model_tags mt ON t.id = mt.tag_id
                    WHERE mt.model_id = ?
                """, (model_id,))
                tags = [r[0] for r in cursor.fetchall()]

                # Get trigger words
                cursor.execute("""
                    SELECT tw.word FROM trigger_words tw
                    JOIN model_trigger_words mtw ON tw.id = mtw.word_id
                    WHERE mtw.model_id = ?
                """, (model_id,))
                trigger_words = [r[0] for r in cursor.fetchall()]

                # Build ModelMetadata (import here to avoid circular import)
                from ml_lib.diffusion.model_metadata import ModelMetadata

                metadata = ModelMetadata(
                    model_id=row["model_id"],
                    name=row["name"],
                    source=Source(row["source"]),
                    type=ModelType(row["type"]),
                    base_model=BaseModel(row["base_model"]),
                    version=row["version"],
                    format=ModelFormat(row["format"]),
                    size_bytes=row["size_bytes"],
                    sha256=row["sha256"] or "",
                    trigger_words=trigger_words,
                    tags=tags,
                    description=row["description"] or "",
                    download_count=row["download_count"],
                    rating=row["rating"],
                    recommended_weight=row["recommended_weight"],
                    local_path=Path(row["local_path"]) if row["local_path"] else None,
                    remote_url=row["remote_url"] or "",
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                )

                # Apply user overrides
                overrides = self.get_user_override(model_id)
                if overrides:
                    for field, value in overrides.items():
                        if hasattr(metadata, field):
                            setattr(metadata, field, value)

                return metadata

        except Exception as e:
            logger.error(f"Failed to get model {model_id}: {e}")
            return None

    def search_models(
        self,
        model_type: Optional[ModelType] = None,
        base_model: Optional[BaseModel] = None,
        tags: Optional[list[str]] = None,
        min_rating: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        """
        Search models with filters.

        Args:
            model_type: Filter by model type
            base_model: Filter by base model
            tags: Filter by tags (ANY match)
            min_rating: Minimum rating
            limit: Max results
            offset: Offset for pagination

        Returns:
            List of matching ModelMetadata
        """
        try:
            with self.connection() as conn:
                cursor = conn.cursor()

                # Build query
                query = "SELECT DISTINCT m.* FROM models m"
                conditions = []
                params = []

                # Join tags if needed
                if tags:
                    query += """
                        JOIN model_tags mt ON m.model_id = mt.model_id
                        JOIN tags t ON mt.tag_id = t.id
                    """
                    placeholders = ",".join("?" * len(tags))
                    conditions.append(f"t.tag IN ({placeholders})")
                    params.extend(tags)

                # Filters
                if model_type:
                    conditions.append("m.type = ?")
                    params.append(model_type.value)

                if base_model:
                    conditions.append("m.base_model = ?")
                    params.append(base_model.value)

                if min_rating > 0:
                    conditions.append("m.rating >= ?")
                    params.append(min_rating)

                # Build WHERE clause
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)

                # Order and pagination
                query += " ORDER BY m.rating DESC, m.download_count DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cursor.execute(query, params)
                rows = cursor.fetchall()

                # Convert to ModelMetadata
                results = []
                for row in rows:
                    metadata = self.get_model(row["model_id"])
                    if metadata:
                        results.append(metadata)

                return results

        except Exception as e:
            logger.error(f"Failed to search models: {e}")
            return []

    def get_all_models_by_type(self, model_type: ModelType) -> list:
        """Get all models of a specific type."""
        return self.search_models(model_type=model_type, limit=10000)

    def delete_model(self, model_id: str) -> bool:
        """Delete model from database (CASCADE deletes tags/trigger words)."""
        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {e}")
            return False

    def get_stats(self) -> dict:
        """Get database statistics."""
        try:
            with self.connection() as conn:
                cursor = conn.cursor()

                stats = {}

                # Count by type
                cursor.execute("SELECT type, COUNT(*) as count FROM models GROUP BY type")
                stats["by_type"] = {row["type"]: row["count"] for row in cursor.fetchall()}

                # Count by base model
                cursor.execute("SELECT base_model, COUNT(*) as count FROM models GROUP BY base_model")
                stats["by_base_model"] = {row["base_model"]: row["count"] for row in cursor.fetchall()}

                # Total models
                cursor.execute("SELECT COUNT(*) as total FROM models")
                stats["total"] = cursor.fetchone()["total"]

                # Total with local path
                cursor.execute("SELECT COUNT(*) as local FROM models WHERE local_path IS NOT NULL")
                stats["local_models"] = cursor.fetchone()["local"]

                # Total with user overrides
                cursor.execute("SELECT COUNT(*) as overrides FROM user_overrides")
                stats["user_overrides"] = cursor.fetchone()["overrides"]

                return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def set_user_override(self, model_id: str, override_fields: dict) -> bool:
        """
        Set user override for model metadata.

        Args:
            model_id: Model identifier
            override_fields: Fields to override (e.g., {"recommended_weight": 0.8, "tags": ["custom"]})

        Returns:
            True if successful
        """
        try:
            import json

            with self.connection() as conn:
                cursor = conn.cursor()

                # Check if model exists
                cursor.execute("SELECT model_id FROM models WHERE model_id = ?", (model_id,))
                if not cursor.fetchone():
                    logger.error(f"Model {model_id} not found")
                    return False

                # Store override as JSON
                override_json = json.dumps(override_fields)

                cursor.execute("""
                    INSERT OR REPLACE INTO user_overrides (model_id, override_data, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (model_id, override_json))

                conn.commit()
                logger.info(f"Set user override for {model_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to set override: {e}")
            return False

    def get_user_override(self, model_id: str) -> Optional[dict]:
        """
        Get user override for model.

        Args:
            model_id: Model identifier

        Returns:
            Override dict or None
        """
        try:
            import json

            with self.connection() as conn:
                cursor = conn.cursor()

                cursor.execute("SELECT override_data FROM user_overrides WHERE model_id = ?", (model_id,))
                row = cursor.fetchone()

                if row:
                    return json.loads(row["override_data"])
                return None

        except Exception as e:
            logger.error(f"Failed to get override: {e}")
            return None

    def delete_user_override(self, model_id: str) -> bool:
        """Delete user override for model."""
        try:
            with self.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM user_overrides WHERE model_id = ?", (model_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to delete override: {e}")
            return False
