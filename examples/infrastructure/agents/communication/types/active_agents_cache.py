"""Active agents cache - NO dictionaries."""

from dataclasses import dataclass, field
from uuid import UUID

from infrastructure.agents.communication.types.active_agents_entry import (
    ActiveAgentsEntry,
)


@dataclass
class ActiveAgentsCache:
    """Type-safe cache for project active agents - NO dictionaries."""

    _entries: list[ActiveAgentsEntry] = field(default_factory=list)

    def add_agent(self, project_id: UUID, agent_id: UUID) -> None:
        """Add agent to project's active agents."""
        # Find existing entry or create new one
        for entry in self._entries:
            if entry.project_id == project_id:
                if agent_id not in entry.agent_ids:
                    entry.agent_ids.append(agent_id)
                return

        # Create new entry
        self._entries.append(ActiveAgentsEntry(project_id, [agent_id]))

    def remove_agent(self, project_id: UUID, agent_id: UUID) -> bool:
        """Remove agent from project's active agents."""
        for entry in self._entries:
            if entry.project_id == project_id:
                if agent_id in entry.agent_ids:
                    entry.agent_ids.remove(agent_id)
                    return True
        return False

    def get_agents(self, project_id: UUID) -> list[UUID]:
        """Get all active agents for a project."""
        for entry in self._entries:
            if entry.project_id == project_id:
                return entry.agent_ids.copy()
        return []

    def remove_project(self, project_id: UUID) -> list[UUID]:
        """Remove project and return its agent IDs."""
        for i, entry in enumerate(self._entries):
            if entry.project_id == project_id:
                agent_ids = entry.agent_ids.copy()
                self._entries.pop(i)
                return agent_ids
        return []

    def has_agent(self, project_id: UUID, agent_id: UUID) -> bool:
        """Check if project has specific agent."""
        for entry in self._entries:
            if entry.project_id == project_id:
                return agent_id in entry.agent_ids
        return False

    def get_agent_count(self, project_id: UUID) -> int:
        """Get number of active agents for project."""
        for entry in self._entries:
            if entry.project_id == project_id:
                return len(entry.agent_ids)
        return 0

    def get_all_project_ids(self) -> list[UUID]:
        """Get all project IDs with active agents."""
        return [entry.project_id for entry in self._entries]

    def clear(self) -> None:
        """Clear all active agents."""
        self._entries.clear()
