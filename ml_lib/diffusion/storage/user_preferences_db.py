"""User preferences database for personalized model selection."""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class UserPreferences:
    """User preferences for generation."""

    user_id: str
    # Model preferences
    favorite_base_models: list[str]  # Model names user prefers
    favorite_loras: list[str]  # LoRA names user prefers
    blocked_models: list[str]  # Models to avoid
    blocked_loras: list[str]  # LoRAs to avoid

    # Generation preferences
    default_quality: str  # "fast", "balanced", "high", "ultra"
    default_steps: int  # Default step count
    default_cfg: float  # Default CFG scale
    default_sampler: str  # Default sampler

    # Content preferences
    preferred_style: Optional[str] = None  # "realistic", "anime", etc.
    nsfw_enabled: bool = True  # Allow NSFW content

    # Advanced preferences
    preferred_resolution: tuple[int, int] = (1024, 1024)  # (width, height)
    memory_mode: str = "auto"  # "auto", "low", "balanced", "aggressive"


class UserPreferencesDB:
    """
    User preferences database.

    Stores user-specific preferences for model selection and generation.
    """

    def __init__(self, db_path: Path | str):
        """
        Initialize database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info(f"UserPreferencesDB initialized: {self.db_path}")

    def _init_schema(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # User preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    default_quality TEXT DEFAULT 'balanced',
                    default_steps INTEGER DEFAULT 30,
                    default_cfg REAL DEFAULT 7.0,
                    default_sampler TEXT DEFAULT 'DPM++ 2M',
                    preferred_style TEXT,
                    nsfw_enabled INTEGER DEFAULT 1,
                    preferred_width INTEGER DEFAULT 1024,
                    preferred_height INTEGER DEFAULT 1024,
                    memory_mode TEXT DEFAULT 'auto',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Favorite models table (many-to-many)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_favorite_models (
                    user_id TEXT,
                    model_name TEXT,
                    model_type TEXT,  -- 'base' or 'lora'
                    priority INTEGER DEFAULT 0,  -- Higher = more preferred
                    times_used INTEGER DEFAULT 0,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, model_name, model_type)
                )
            """)

            # Blocked models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_blocked_models (
                    user_id TEXT,
                    model_name TEXT,
                    model_type TEXT,  -- 'base' or 'lora'
                    reason TEXT,
                    blocked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, model_name, model_type)
                )
            """)

            # Generation history (for learning preferences)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generation_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    prompt_hash TEXT,  -- Hash of prompt for deduplication
                    base_model TEXT,
                    loras TEXT,  -- JSON array of LoRA names
                    quality TEXT,
                    steps INTEGER,
                    cfg REAL,
                    sampler TEXT,
                    width INTEGER,
                    height INTEGER,
                    rating INTEGER,  -- User rating 1-5 (nullable)
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_generation_history_user
                ON generation_history(user_id)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_generation_history_rating
                ON generation_history(user_id, rating)
            """)

            conn.commit()

        logger.info("Database schema initialized")

    def get_or_create_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences or create defaults."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Check if exists
            cursor.execute(
                "SELECT * FROM user_preferences WHERE user_id = ?",
                (user_id,)
            )
            row = cursor.fetchone()

            if row:
                # Parse existing preferences
                return self._row_to_preferences(row, user_id)
            else:
                # Create defaults
                cursor.execute("""
                    INSERT INTO user_preferences (user_id) VALUES (?)
                """, (user_id,))
                conn.commit()

                return self.get_or_create_preferences(user_id)

    def _row_to_preferences(self, row, user_id: str) -> UserPreferences:
        """Convert database row to UserPreferences."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            # Get favorite models
            cursor.execute("""
                SELECT model_name FROM user_favorite_models
                WHERE user_id = ? AND model_type = 'base'
                ORDER BY priority DESC, times_used DESC
            """, (user_id,))
            favorite_base_models = [r[0] for r in cursor.fetchall()]

            # Get favorite LoRAs
            cursor.execute("""
                SELECT model_name FROM user_favorite_models
                WHERE user_id = ? AND model_type = 'lora'
                ORDER BY priority DESC, times_used DESC
            """, (user_id,))
            favorite_loras = [r[0] for r in cursor.fetchall()]

            # Get blocked models
            cursor.execute("""
                SELECT model_name FROM user_blocked_models
                WHERE user_id = ? AND model_type = 'base'
            """, (user_id,))
            blocked_models = [r[0] for r in cursor.fetchall()]

            # Get blocked LoRAs
            cursor.execute("""
                SELECT model_name FROM user_blocked_models
                WHERE user_id = ? AND model_type = 'lora'
            """, (user_id,))
            blocked_loras = [r[0] for r in cursor.fetchall()]

        return UserPreferences(
            user_id=user_id,
            favorite_base_models=favorite_base_models,
            favorite_loras=favorite_loras,
            blocked_models=blocked_models,
            blocked_loras=blocked_loras,
            default_quality=row[1],
            default_steps=row[2],
            default_cfg=row[3],
            default_sampler=row[4],
            preferred_style=row[5],
            nsfw_enabled=bool(row[6]),
            preferred_resolution=(row[7], row[8]),
            memory_mode=row[9],
        )

    def update_preferences(self, prefs: UserPreferences):
        """Update user preferences."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                UPDATE user_preferences SET
                    default_quality = ?,
                    default_steps = ?,
                    default_cfg = ?,
                    default_sampler = ?,
                    preferred_style = ?,
                    nsfw_enabled = ?,
                    preferred_width = ?,
                    preferred_height = ?,
                    memory_mode = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (
                prefs.default_quality,
                prefs.default_steps,
                prefs.default_cfg,
                prefs.default_sampler,
                prefs.preferred_style,
                1 if prefs.nsfw_enabled else 0,
                prefs.preferred_resolution[0],
                prefs.preferred_resolution[1],
                prefs.memory_mode,
                prefs.user_id,
            ))

            conn.commit()

        logger.info(f"Updated preferences for user: {prefs.user_id}")

    def add_favorite_model(self, user_id: str, model_name: str, model_type: str, priority: int = 5):
        """Add model to favorites."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO user_favorite_models
                (user_id, model_name, model_type, priority)
                VALUES (?, ?, ?, ?)
            """, (user_id, model_name, model_type, priority))

            conn.commit()

        logger.info(f"Added favorite {model_type}: {model_name} for user {user_id}")

    def block_model(self, user_id: str, model_name: str, model_type: str, reason: str = ""):
        """Block a model from being used."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT OR REPLACE INTO user_blocked_models
                (user_id, model_name, model_type, reason)
                VALUES (?, ?, ?, ?)
            """, (user_id, model_name, model_type, reason))

            conn.commit()

        logger.info(f"Blocked {model_type}: {model_name} for user {user_id}")

    def record_generation(
        self,
        user_id: str,
        prompt: str,
        base_model: str,
        loras: list[str],
        quality: str,
        steps: int,
        cfg: float,
        sampler: str,
        width: int,
        height: int,
        rating: Optional[int] = None,
    ):
        """Record a generation for learning preferences."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()

        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO generation_history
                (user_id, prompt_hash, base_model, loras, quality, steps, cfg, sampler, width, height, rating)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                prompt_hash,
                base_model,
                json.dumps(loras),
                quality,
                steps,
                cfg,
                sampler,
                width,
                height,
                rating,
            ))

            # Update usage count for models
            cursor.execute("""
                INSERT OR IGNORE INTO user_favorite_models
                (user_id, model_name, model_type, times_used)
                VALUES (?, ?, 'base', 0)
            """, (user_id, base_model))

            cursor.execute("""
                UPDATE user_favorite_models
                SET times_used = times_used + 1, last_used = CURRENT_TIMESTAMP
                WHERE user_id = ? AND model_name = ? AND model_type = 'base'
            """, (user_id, base_model))

            # Same for LoRAs
            for lora in loras:
                cursor.execute("""
                    INSERT OR IGNORE INTO user_favorite_models
                    (user_id, model_name, model_type, times_used)
                    VALUES (?, ?, 'lora', 0)
                """, (user_id, lora))

                cursor.execute("""
                    UPDATE user_favorite_models
                    SET times_used = times_used + 1, last_used = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND model_name = ? AND model_type = 'lora'
                """, (user_id, lora))

            conn.commit()

    def get_most_used_models(self, user_id: str, model_type: str, limit: int = 10) -> list[str]:
        """Get most used models for a user."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT model_name, times_used
                FROM user_favorite_models
                WHERE user_id = ? AND model_type = ?
                ORDER BY times_used DESC, last_used DESC
                LIMIT ?
            """, (user_id, model_type, limit))

            return [row[0] for row in cursor.fetchall()]
