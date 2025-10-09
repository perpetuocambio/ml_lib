"""Agent state management using PostgreSQL."""

from datetime import datetime
from uuid import UUID, uuid4

import asyncpg


class AgentStateManager:
    """Manages agent state persistence in PostgreSQL."""

    def __init__(self, connection_pool: asyncpg.Pool):
        """Initialize with database connection pool."""
        self.pool = connection_pool

    async def create_agent_state(
        self, project_id: UUID, agent_type: str, initial_state: str
    ) -> UUID:
        """Create new agent state record."""
        agent_id = uuid4()

        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO agent_states (agent_id, project_id, agent_type, state, created_at)
                VALUES ($1, $2, $3, $4, $5)
                """,
                agent_id,
                project_id,
                agent_type,
                initial_state,
                datetime.utcnow(),
            )

        return agent_id

    async def get_agent_state(self, agent_id: UUID) -> str | None:
        """Retrieve agent state by ID."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT state FROM agent_states WHERE agent_id = $1", agent_id
            )
            return row["state"] if row else None

    async def update_agent_state(self, agent_id: UUID, new_state: str) -> bool:
        """Update agent state."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE agent_states
                SET state = $1, updated_at = $2
                WHERE agent_id = $3
                """,
                new_state,
                datetime.utcnow(),
                agent_id,
            )
            return result != "UPDATE 0"

    async def delete_agent_state(self, agent_id: UUID) -> bool:
        """Delete agent state."""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM agent_states WHERE agent_id = $1", agent_id
            )
            return result != "DELETE 0"
