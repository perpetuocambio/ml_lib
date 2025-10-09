"""SQLite implementation of prompt version repository."""

import sqlite3

from infrastructure.persistence.agents.prompt_version_data import PromptVersionData


class SQLitePromptVersionRepository:
    """SQLite implementation of prompt version repository."""

    def __init__(self, db_path: str):
        """Initialize repository with database path."""
        self.db_path = db_path
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        """Create prompt_versions table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prompt_versions (
                    version_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    prompt_content TEXT NOT NULL,
                    version_number INTEGER NOT NULL,
                    change_description TEXT,
                    created_at TEXT NOT NULL,
                    is_active BOOLEAN NOT NULL DEFAULT FALSE,
                    UNIQUE(agent_id, version_number)
                )
            """)
            conn.commit()

    async def save_version(self, version_data: PromptVersionData) -> None:
        """Save a new prompt version."""
        with sqlite3.connect(self.db_path) as conn:
            # If this version should be active, deactivate others for this agent
            if version_data.is_active:
                conn.execute(
                    "UPDATE prompt_versions SET is_active = FALSE WHERE agent_id = ?",
                    (version_data.agent_id,),
                )

            conn.execute(
                """
                INSERT INTO prompt_versions
                (version_id, agent_id, prompt_content, version_number,
                 change_description, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    version_data.version_id,
                    version_data.agent_id,
                    version_data.prompt_content,
                    version_data.version_number,
                    version_data.change_description,
                    version_data.created_at,
                    version_data.is_active,
                ),
            )
            conn.commit()

    async def get_active_version(self, agent_id: str) -> PromptVersionData | None:
        """Get the currently active prompt version for an agent."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT version_id, agent_id, prompt_content, version_number,
                       change_description, created_at, is_active
                FROM prompt_versions
                WHERE agent_id = ? AND is_active = TRUE
            """,
                (agent_id,),
            )

            row = cursor.fetchone()
            if row:
                return PromptVersionData(
                    version_id=row[0],
                    agent_id=row[1],
                    prompt_content=row[2],
                    version_number=row[3],
                    change_description=row[4],
                    created_at=row[5],
                    is_active=bool(row[6]),
                )
            return None

    async def get_version_history(self, agent_id: str) -> list[PromptVersionData]:
        """Get all prompt versions for an agent, ordered by version number."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT version_id, agent_id, prompt_content, version_number,
                       change_description, created_at, is_active
                FROM prompt_versions
                WHERE agent_id = ?
                ORDER BY version_number DESC
            """,
                (agent_id,),
            )

            versions = []
            for row in cursor.fetchall():
                versions.append(
                    PromptVersionData(
                        version_id=row[0],
                        agent_id=row[1],
                        prompt_content=row[2],
                        version_number=row[3],
                        change_description=row[4],
                        created_at=row[5],
                        is_active=bool(row[6]),
                    )
                )
            return versions

    async def get_version_by_id(self, version_id: str) -> PromptVersionData | None:
        """Get a specific prompt version by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT version_id, agent_id, prompt_content, version_number,
                       change_description, created_at, is_active
                FROM prompt_versions
                WHERE version_id = ?
            """,
                (version_id,),
            )

            row = cursor.fetchone()
            if row:
                return PromptVersionData(
                    version_id=row[0],
                    agent_id=row[1],
                    prompt_content=row[2],
                    version_number=row[3],
                    change_description=row[4],
                    created_at=row[5],
                    is_active=bool(row[6]),
                )
            return None

    async def activate_version(self, agent_id: str, version_id: str) -> bool:
        """Set a specific version as active and deactivate others."""
        with sqlite3.connect(self.db_path) as conn:
            # First, check if the version exists
            cursor = conn.execute(
                "SELECT COUNT(*) FROM prompt_versions WHERE version_id = ? AND agent_id = ?",
                (version_id, agent_id),
            )
            if cursor.fetchone()[0] == 0:
                return False

            # Deactivate all versions for this agent
            conn.execute(
                "UPDATE prompt_versions SET is_active = FALSE WHERE agent_id = ?",
                (agent_id,),
            )

            # Activate the specific version
            cursor = conn.execute(
                "UPDATE prompt_versions SET is_active = TRUE WHERE version_id = ?",
                (version_id,),
            )
            conn.commit()

            return cursor.rowcount > 0

    async def delete_version(self, version_id: str) -> bool:
        """Delete a prompt version. Cannot delete if it's the only version."""
        with sqlite3.connect(self.db_path) as conn:
            # Get agent_id for this version
            cursor = conn.execute(
                "SELECT agent_id FROM prompt_versions WHERE version_id = ?",
                (version_id,),
            )
            row = cursor.fetchone()
            if not row:
                return False

            agent_id = row[0]

            # Check if this is the only version for the agent
            cursor = conn.execute(
                "SELECT COUNT(*) FROM prompt_versions WHERE agent_id = ?", (agent_id,)
            )
            if cursor.fetchone()[0] <= 1:
                return False  # Cannot delete the only version

            # Delete the version
            cursor = conn.execute(
                "DELETE FROM prompt_versions WHERE version_id = ?", (version_id,)
            )
            conn.commit()

            return cursor.rowcount > 0
