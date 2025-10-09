"""SQLite implementation of infrastructure agent persistence."""

import json
import sqlite3
from datetime import datetime

from infrastructure.persistence.agents.agent_data import AgentData
from infrastructure.persistence.agents.interfaces.agent_persistence_interface import (
    IAgentPersistence,
)


class SQLiteAgentPersistence(IAgentPersistence):
    """Infrastructure agent repository - no external dependencies."""

    def __init__(self, db_path: str):
        """Initialize repository with database path."""
        self.db_path = db_path
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        """Create agents table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    role_name TEXT NOT NULL,
                    role_description TEXT NOT NULL,
                    system_prompt TEXT NOT NULL,
                    preferred_tools TEXT NOT NULL,  -- JSON array
                    expertise_domains TEXT NOT NULL,  -- JSON array
                    reasoning_style TEXT NOT NULL,
                    autonomy_level TEXT NOT NULL,
                    collaboration_style TEXT NOT NULL,
                    confidence_threshold REAL NOT NULL,
                    max_concurrent_tasks INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    last_active_at TEXT NOT NULL,
                    project_id TEXT NOT NULL,
                    user_id TEXT NOT NULL
                )
            """)
            conn.commit()

    def save_agent(self, agent_data: AgentData) -> None:
        """Save or update an agent."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agents
                (agent_id, name, role_name, role_description, system_prompt,
                 preferred_tools, expertise_domains, reasoning_style,
                 autonomy_level, collaboration_style, confidence_threshold,
                 max_concurrent_tasks, status, created_at, last_active_at,
                 project_id, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    agent_data.agent_id,
                    agent_data.name,
                    agent_data.role_name,
                    agent_data.role_description,
                    agent_data.system_prompt,
                    json.dumps(agent_data.preferred_tools),
                    json.dumps(agent_data.expertise_domains),
                    agent_data.reasoning_style,
                    agent_data.autonomy_level,
                    agent_data.collaboration_style,
                    agent_data.confidence_threshold,
                    agent_data.max_concurrent_tasks,
                    agent_data.status,
                    agent_data.created_at.isoformat(),
                    agent_data.last_active_at.isoformat(),
                    agent_data.project_id,
                    agent_data.user_id,
                ),
            )
            conn.commit()

    def get_agent_by_id(self, agent_id: str) -> AgentData | None:
        """Get agent by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM agents WHERE agent_id = ?", (agent_id,)
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_agent_data(row)
            return None

    def get_agents_by_project_id(self, project_id: str) -> list[AgentData]:
        """Get all agents for a specific project."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM agents WHERE project_id = ?", (project_id,)
            )
            return [self._row_to_agent_data(row) for row in cursor.fetchall()]

    def get_agents_by_user_id(self, user_id: str) -> list[AgentData]:
        """Get all agents for a specific user."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM agents WHERE user_id = ?", (user_id,))
            return [self._row_to_agent_data(row) for row in cursor.fetchall()]

    def delete_agent(self, agent_id: str) -> bool:
        """Delete agent by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM agents WHERE agent_id = ?", (agent_id,))
            conn.commit()
            return cursor.rowcount > 0

    def agent_exists(self, agent_id: str) -> bool:
        """Check if agent exists."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM agents WHERE agent_id = ?", (agent_id,)
            )
            return cursor.fetchone()[0] > 0

    def _row_to_agent_data(self, row: tuple) -> AgentData:
        """Convert database row to AgentData DTO."""
        (
            agent_id_str,
            name,
            role_name,
            role_description,
            system_prompt,
            preferred_tools_json,
            expertise_domains_json,
            reasoning_style_str,
            autonomy_level_str,
            collaboration_style,
            confidence_threshold,
            max_concurrent_tasks,
            status_str,
            created_at_str,
            last_active_at_str,
            project_id,
            user_id,
        ) = row

        # Parse JSON fields
        preferred_tools = json.loads(preferred_tools_json)
        expertise_domains = json.loads(expertise_domains_json)

        return AgentData(
            agent_id=agent_id_str,
            name=name,
            role_name=role_name,
            role_description=role_description,
            system_prompt=system_prompt,
            preferred_tools=preferred_tools,
            expertise_domains=expertise_domains,
            reasoning_style=reasoning_style_str,
            autonomy_level=autonomy_level_str,
            collaboration_style=collaboration_style,
            confidence_threshold=confidence_threshold,
            max_concurrent_tasks=max_concurrent_tasks,
            status=status_str,
            created_at=datetime.fromisoformat(created_at_str),
            last_active_at=datetime.fromisoformat(last_active_at_str),
            project_id=project_id,
            user_id=user_id,
        )
