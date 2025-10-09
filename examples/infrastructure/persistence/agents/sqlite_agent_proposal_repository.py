"""SQLite implementation of agent proposal repository."""

import sqlite3
from datetime import datetime

from infrastructure.persistence.agents.agent_proposal_data import AgentProposalData


class SQLiteAgentProposalRepository:
    """SQLite implementation of agent proposal repository."""

    def __init__(self, db_path: str):
        """Initialize repository with database path."""
        self.db_path = db_path
        self._create_table_if_not_exists()

    def _create_table_if_not_exists(self) -> None:
        """Create agent_proposals table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    justification TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    priority TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_arguments TEXT NOT NULL,
                    expected_outcome TEXT NOT NULL,
                    estimated_duration_minutes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    approved_at TEXT,
                    approved_by_user_id TEXT,
                    rejection_reason TEXT
                )
            """)
            conn.commit()

    async def save(self, proposal_data: AgentProposalData) -> None:
        """Save or update an agent proposal."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO agent_proposals
                (proposal_id, agent_id, title, description, justification,
                 confidence, priority, status, tool_name, tool_arguments,
                 expected_outcome, estimated_duration_minutes, created_at,
                 approved_at, approved_by_user_id, rejection_reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    proposal_data.proposal_id,
                    proposal_data.agent_id,
                    proposal_data.title,
                    proposal_data.description,
                    proposal_data.justification,
                    proposal_data.confidence,
                    proposal_data.priority,
                    proposal_data.status,
                    proposal_data.tool_name,
                    proposal_data.tool_arguments,
                    proposal_data.expected_outcome,
                    proposal_data.estimated_duration_minutes,
                    proposal_data.created_at.isoformat(),
                    proposal_data.approved_at.isoformat()
                    if proposal_data.approved_at
                    else None,
                    proposal_data.approved_by_user_id,
                    proposal_data.rejection_reason,
                ),
            )
            conn.commit()

    async def get_by_id(self, proposal_id: str) -> AgentProposalData | None:
        """Get proposal by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM agent_proposals WHERE proposal_id = ?",
                (proposal_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_proposal_data(row)
            return None

    async def get_by_agent_id(self, agent_id: str) -> list[AgentProposalData]:
        """Get all proposals for a specific agent."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM agent_proposals WHERE agent_id = ? ORDER BY created_at DESC",
                (agent_id,),
            )
            return [self._row_to_proposal_data(row) for row in cursor.fetchall()]

    async def get_by_status(self, status: str) -> list[AgentProposalData]:
        """Get all proposals with a specific status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM agent_proposals WHERE status = ? ORDER BY created_at DESC",
                (status,),
            )
            return [self._row_to_proposal_data(row) for row in cursor.fetchall()]

    async def get_pending_approval(self) -> list[AgentProposalData]:
        """Get all proposals pending approval."""
        return await self.get_by_status("PENDING")

    async def delete(self, proposal_id: str) -> bool:
        """Delete proposal by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM agent_proposals WHERE proposal_id = ?",
                (proposal_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    async def count_active_proposals_for_agent(self, agent_id: str) -> int:
        """Count active proposals for an agent."""
        active_statuses = [
            "PENDING",
            "APPROVED",
            "AUTO_APPROVED",
            "EXECUTING",
        ]

        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join("?" * len(active_statuses))
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM agent_proposals WHERE agent_id = ? AND status IN ({placeholders})",
                (agent_id, *active_statuses),
            )
            return cursor.fetchone()[0]

    def _row_to_proposal_data(self, row: tuple) -> AgentProposalData:
        """Convert database row to AgentProposalData DTO."""
        (
            proposal_id_str,
            agent_id_str,
            title,
            description,
            justification,
            confidence,
            priority_str,
            status_str,
            tool_name,
            tool_arguments,
            expected_outcome,
            estimated_duration_minutes,
            created_at_str,
            approved_at_str,
            approved_by_user_id,
            rejection_reason,
        ) = row

        return AgentProposalData(
            proposal_id=proposal_id_str,
            agent_id=agent_id_str,
            title=title,
            description=description,
            justification=justification,
            confidence=confidence,
            priority=priority_str,
            status=status_str,
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            expected_outcome=expected_outcome,
            estimated_duration_minutes=estimated_duration_minutes,
            created_at=datetime.fromisoformat(created_at_str),
            approved_at=datetime.fromisoformat(approved_at_str)
            if approved_at_str
            else None,
            approved_by_user_id=approved_by_user_id,
            rejection_reason=rejection_reason,
        )
