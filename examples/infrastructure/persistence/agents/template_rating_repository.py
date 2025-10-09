"""Infrastructure template rating repository - completely isolated implementation."""

import sqlite3

from infrastructure.persistence.agents.template_rating_data import TemplateRatingData
from infrastructure.persistence.agents.template_rating_statistics_data import (
    TemplateRatingStatisticsData,
)


class InfrastructureTemplateRatingRepository:
    """Infrastructure template rating repository - no external dependencies."""

    def __init__(self, db_path: str):
        """Initialize repository with database path."""
        self.db_path = db_path
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        """Create template_ratings table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS template_ratings (
                    rating_id TEXT PRIMARY KEY,
                    template_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    review_text TEXT,
                    created_at TEXT NOT NULL,
                    is_verified_user BOOLEAN NOT NULL DEFAULT FALSE,
                    UNIQUE(template_id, user_id)
                )
            """)
            conn.commit()

    async def save_rating(self, rating_data: TemplateRatingData) -> None:
        """Save a new template rating."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO template_ratings
                (rating_id, template_id, user_id, rating, review_text,
                 created_at, is_verified_user)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    rating_data.rating_id,
                    rating_data.template_id,
                    rating_data.user_id,
                    rating_data.rating,
                    rating_data.review_text,
                    rating_data.created_at,
                    rating_data.is_verified_user,
                ),
            )
            conn.commit()

    async def get_user_rating(
        self, template_id: str, user_id: str
    ) -> TemplateRatingData | None:
        """Get a specific user's rating for a template."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT rating_id, template_id, user_id, rating, review_text,
                       created_at, is_verified_user
                FROM template_ratings
                WHERE template_id = ? AND user_id = ?
            """,
                (template_id, user_id),
            )

            row = cursor.fetchone()
            if row:
                return TemplateRatingData(
                    rating_id=row[0],
                    template_id=row[1],
                    user_id=row[2],
                    rating=row[3],
                    review_text=row[4],
                    created_at=row[5],
                    is_verified_user=bool(row[6]),
                )
            return None

    async def get_template_ratings(self, template_id: str) -> list[TemplateRatingData]:
        """Get all ratings for a specific template."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT rating_id, template_id, user_id, rating, review_text,
                       created_at, is_verified_user
                FROM template_ratings
                WHERE template_id = ?
                ORDER BY created_at DESC
            """,
                (template_id,),
            )

            ratings = []
            for row in cursor.fetchall():
                ratings.append(
                    TemplateRatingData(
                        rating_id=row[0],
                        template_id=row[1],
                        user_id=row[2],
                        rating=row[3],
                        review_text=row[4],
                        created_at=row[5],
                        is_verified_user=bool(row[6]),
                    )
                )
            return ratings

    async def get_template_rating_stats(
        self, template_id: str
    ) -> TemplateRatingStatisticsData:
        """Get rating statistics for a template (avg, count, distribution)."""
        with sqlite3.connect(self.db_path) as conn:
            # Get basic stats
            cursor = conn.execute(
                """
                SELECT
                    AVG(CAST(rating AS FLOAT)) as avg_rating,
                    COUNT(*) as total_count,
                    COUNT(CASE WHEN is_verified_user = 1 THEN 1 END) as verified_count
                FROM template_ratings
                WHERE template_id = ?
            """,
                (template_id,),
            )

            row = cursor.fetchone()
            if not row or row[0] is None:
                return TemplateRatingStatisticsData(
                    avg_rating=0.0,
                    total_count=0,
                    verified_count=0,
                )

            # Get rating distribution
            cursor = conn.execute(
                """
                SELECT rating, COUNT(*) as count
                FROM template_ratings
                WHERE template_id = ?
                GROUP BY rating
                ORDER BY rating
            """,
                (template_id,),
            )

            # Initialize distribution counts
            star_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for rating_row in cursor.fetchall():
                star_counts[rating_row[0]] = rating_row[1]

            return TemplateRatingStatisticsData(
                avg_rating=float(row[0]) if row[0] else 0.0,
                total_count=int(row[1]),
                verified_count=int(row[2]),
                stars_1=star_counts[1],
                stars_2=star_counts[2],
                stars_3=star_counts[3],
                stars_4=star_counts[4],
                stars_5=star_counts[5],
            )

    async def update_rating(self, rating_data: TemplateRatingData) -> bool:
        """Update an existing rating."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE template_ratings
                SET rating = ?, review_text = ?, created_at = ?, is_verified_user = ?
                WHERE rating_id = ?
            """,
                (
                    rating_data.rating,
                    rating_data.review_text,
                    rating_data.created_at,
                    rating_data.is_verified_user,
                    rating_data.rating_id,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    async def delete_rating(self, rating_id: str) -> bool:
        """Delete a rating."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM template_ratings WHERE rating_id = ?", (rating_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
