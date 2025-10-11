"""Learning engine for continuous improvement from user feedback."""

import logging
import sqlite3
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import json

from ml_lib.diffusion.intelligent.prompting.entities import (
    PromptAnalysis,
    LoRARecommendation,
    OptimizedParameters,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationFeedback:
    """Feedback for a generation."""

    feedback_id: str
    timestamp: str
    original_prompt: str

    # What was recommended
    recommended_loras: list[str]
    recommended_params: dict

    # User rating (1-5)
    rating: int

    # What user changed (if any)
    user_modified_loras: Optional[list[str]] = None
    user_modified_params: Optional[dict] = None

    # Tags/notes
    tags: list[str] = None
    notes: str = ""

    def __post_init__(self):
        """Validate feedback."""
        assert 1 <= self.rating <= 5, "Rating must be between 1 and 5"
        if self.tags is None:
            self.tags = []


class LearningEngine:
    """
    Learning engine that improves recommendations based on user feedback.

    Tracks:
    - Which LoRAs users prefer for specific concepts
    - Which parameters users adjust
    - Success rates of recommendations
    - Common patterns in user modifications

    Uses:
    - SQLite database for persistence
    - Simple scoring adjustments based on feedback
    - Pattern recognition for common fixes

    Example:
        >>> engine = LearningEngine()
        >>> feedback = GenerationFeedback(
        ...     feedback_id="gen_001",
        ...     timestamp="2025-10-11T12:00:00",
        ...     original_prompt="anime girl with cat ears",
        ...     recommended_loras=["anime_style", "neko_lora"],
        ...     recommended_params={"steps": 30, "cfg": 7.5},
        ...     rating=5
        ... )
        >>> engine.record_feedback(feedback)
        >>> insights = engine.get_insights()
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize learning engine.

        Args:
            db_path: Path to SQLite database (default: ~/.ml_lib/feedback.db)
        """
        if db_path is None:
            db_path = Path.home() / ".ml_lib" / "feedback.db"

        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_database()

        logger.info(f"LearningEngine initialized with database: {self.db_path}")

    def _init_database(self):
        """Initialize SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    original_prompt TEXT NOT NULL,
                    recommended_loras TEXT,
                    recommended_params TEXT,
                    rating INTEGER NOT NULL,
                    user_modified_loras TEXT,
                    user_modified_params TEXT,
                    tags TEXT,
                    notes TEXT
                )
            """)

            # LoRA performance tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS lora_performance (
                    lora_name TEXT PRIMARY KEY,
                    total_recommendations INTEGER DEFAULT 0,
                    total_positive_feedback INTEGER DEFAULT 0,
                    total_negative_feedback INTEGER DEFAULT 0,
                    average_rating REAL DEFAULT 0.0,
                    last_updated TEXT
                )
            """)

            # Concept-LoRA associations (what works for what)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS concept_lora_mapping (
                    concept TEXT,
                    lora_name TEXT,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.5,
                    PRIMARY KEY (concept, lora_name)
                )
            """)

            # Parameter adjustments tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS parameter_adjustments (
                    adjustment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    original_prompt TEXT,
                    parameter_name TEXT NOT NULL,
                    recommended_value REAL,
                    user_value REAL,
                    difference REAL,
                    rating INTEGER
                )
            """)

            conn.commit()

        logger.info("Database schema initialized")

    def record_feedback(self, feedback: GenerationFeedback):
        """
        Record user feedback for a generation.

        Args:
            feedback: Generation feedback data
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Insert feedback
            cursor.execute("""
                INSERT OR REPLACE INTO feedback
                (feedback_id, timestamp, original_prompt, recommended_loras,
                 recommended_params, rating, user_modified_loras,
                 user_modified_params, tags, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.feedback_id,
                feedback.timestamp,
                feedback.original_prompt,
                json.dumps(feedback.recommended_loras),
                json.dumps(feedback.recommended_params),
                feedback.rating,
                json.dumps(feedback.user_modified_loras) if feedback.user_modified_loras else None,
                json.dumps(feedback.user_modified_params) if feedback.user_modified_params else None,
                json.dumps(feedback.tags),
                feedback.notes,
            ))

            # Update LoRA performance
            is_positive = feedback.rating >= 4
            is_negative = feedback.rating <= 2

            for lora in feedback.recommended_loras:
                self._update_lora_performance(
                    cursor, lora, feedback.rating, is_positive, is_negative
                )

            # Track parameter adjustments if user modified
            if feedback.user_modified_params:
                self._track_parameter_adjustments(
                    cursor, feedback
                )

            conn.commit()

        logger.info(f"Recorded feedback {feedback.feedback_id} (rating: {feedback.rating})")

    def _update_lora_performance(
        self, cursor, lora_name: str, rating: int, is_positive: bool, is_negative: bool
    ):
        """Update LoRA performance statistics."""
        now = datetime.now().isoformat()

        # Get current stats
        cursor.execute(
            "SELECT total_recommendations, total_positive_feedback, "
            "total_negative_feedback, average_rating FROM lora_performance WHERE lora_name = ?",
            (lora_name,)
        )
        row = cursor.fetchone()

        if row:
            total_recs, pos, neg, avg_rating = row

            # Update counts
            new_total = total_recs + 1
            new_pos = pos + (1 if is_positive else 0)
            new_neg = neg + (1 if is_negative else 0)

            # Update average rating (running average)
            new_avg = (avg_rating * total_recs + rating) / new_total

            cursor.execute("""
                UPDATE lora_performance
                SET total_recommendations = ?,
                    total_positive_feedback = ?,
                    total_negative_feedback = ?,
                    average_rating = ?,
                    last_updated = ?
                WHERE lora_name = ?
            """, (new_total, new_pos, new_neg, new_avg, now, lora_name))
        else:
            # Insert new entry
            cursor.execute("""
                INSERT INTO lora_performance
                (lora_name, total_recommendations, total_positive_feedback,
                 total_negative_feedback, average_rating, last_updated)
                VALUES (?, 1, ?, ?, ?, ?)
            """, (
                lora_name,
                1 if is_positive else 0,
                1 if is_negative else 0,
                float(rating),
                now
            ))

    def _track_parameter_adjustments(self, cursor, feedback: GenerationFeedback):
        """Track which parameters users commonly adjust."""
        if not feedback.user_modified_params:
            return

        now = datetime.now().isoformat()

        for param_name, user_value in feedback.user_modified_params.items():
            recommended_value = feedback.recommended_params.get(param_name)

            if recommended_value is not None:
                try:
                    rec_val = float(recommended_value)
                    usr_val = float(user_value)
                    difference = usr_val - rec_val

                    cursor.execute("""
                        INSERT INTO parameter_adjustments
                        (timestamp, original_prompt, parameter_name,
                         recommended_value, user_value, difference, rating)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        now,
                        feedback.original_prompt,
                        param_name,
                        rec_val,
                        usr_val,
                        difference,
                        feedback.rating
                    ))
                except (ValueError, TypeError):
                    # Skip non-numeric parameters
                    pass

    def get_lora_adjustment_factor(self, lora_name: str) -> float:
        """
        Get confidence adjustment factor for a LoRA based on past performance.

        Args:
            lora_name: LoRA name

        Returns:
            Adjustment factor (0.5 to 1.5)
            - 1.0 = no adjustment (neutral)
            - >1.0 = boost confidence (performs well)
            - <1.0 = reduce confidence (performs poorly)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT total_recommendations, total_positive_feedback,
                       total_negative_feedback, average_rating
                FROM lora_performance
                WHERE lora_name = ?
            """, (lora_name,))

            row = cursor.fetchone()

            if not row or row[0] < 3:
                # Not enough data, neutral
                return 1.0

            total, positive, negative, avg_rating = row

            # Calculate success rate
            success_rate = positive / total if total > 0 else 0.5

            # Adjustment factor based on success rate and rating
            # success_rate: 0.0 to 1.0
            # avg_rating: 1.0 to 5.0 (normalize to 0.0 to 1.0)
            normalized_rating = (avg_rating - 1.0) / 4.0

            # Combined score (weighted average)
            combined = 0.6 * success_rate + 0.4 * normalized_rating

            # Map to adjustment factor: 0.5 to 1.5
            # 0.0 combined → 0.5 adjustment
            # 0.5 combined → 1.0 adjustment
            # 1.0 combined → 1.5 adjustment
            adjustment = 0.5 + combined

            return round(adjustment, 2)

    def get_parameter_bias(self, param_name: str) -> Optional[float]:
        """
        Get common user adjustment bias for a parameter.

        Args:
            param_name: Parameter name (e.g., "num_steps", "guidance_scale")

        Returns:
            Average adjustment value (None if insufficient data)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get adjustments for well-rated generations only (rating >= 4)
            cursor.execute("""
                SELECT AVG(difference)
                FROM parameter_adjustments
                WHERE parameter_name = ? AND rating >= 4
            """, (param_name,))

            result = cursor.fetchone()

            if result and result[0] is not None:
                return round(result[0], 2)

            return None

    def get_insights(self) -> dict:
        """
        Get learning insights and statistics.

        Returns:
            Dictionary with insights about recommendations and user behavior
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total feedback count
            cursor.execute("SELECT COUNT(*) FROM feedback")
            total_feedback = cursor.fetchone()[0]

            # Average rating
            cursor.execute("SELECT AVG(rating) FROM feedback")
            avg_rating = cursor.fetchone()[0] or 0.0

            # Top performing LoRAs
            cursor.execute("""
                SELECT lora_name, average_rating, total_recommendations
                FROM lora_performance
                WHERE total_recommendations >= 3
                ORDER BY average_rating DESC
                LIMIT 10
            """)
            top_loras = [
                {
                    "name": row[0],
                    "avg_rating": round(row[1], 2),
                    "count": row[2]
                }
                for row in cursor.fetchall()
            ]

            # Most adjusted parameters
            cursor.execute("""
                SELECT parameter_name, AVG(difference), COUNT(*)
                FROM parameter_adjustments
                WHERE rating >= 4
                GROUP BY parameter_name
                HAVING COUNT(*) >= 3
                ORDER BY COUNT(*) DESC
                LIMIT 5
            """)
            common_adjustments = [
                {
                    "parameter": row[0],
                    "avg_adjustment": round(row[1], 2),
                    "count": row[2]
                }
                for row in cursor.fetchall()
            ]

            return {
                "total_feedback_records": total_feedback,
                "overall_average_rating": round(avg_rating, 2),
                "top_performing_loras": top_loras,
                "common_parameter_adjustments": common_adjustments,
            }

    def clear_data(self, older_than_days: Optional[int] = None):
        """
        Clear feedback data.

        Args:
            older_than_days: Only clear data older than N days (None = clear all)
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if older_than_days is None:
                # Clear all
                cursor.execute("DELETE FROM feedback")
                cursor.execute("DELETE FROM lora_performance")
                cursor.execute("DELETE FROM concept_lora_mapping")
                cursor.execute("DELETE FROM parameter_adjustments")
                logger.info("Cleared all feedback data")
            else:
                # Clear old data
                from datetime import timedelta
                cutoff = (datetime.now() - timedelta(days=older_than_days)).isoformat()

                cursor.execute("DELETE FROM feedback WHERE timestamp < ?", (cutoff,))
                cursor.execute("DELETE FROM parameter_adjustments WHERE timestamp < ?", (cutoff,))
                logger.info(f"Cleared feedback data older than {older_than_days} days")

            conn.commit()
