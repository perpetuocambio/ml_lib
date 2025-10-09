"""Active agent cache - NO dictionaries."""

from dataclasses import dataclass, field
from uuid import UUID

from infrastructure.agents.types.active_agent_entry import ActiveAgentEntry
from langgraph.graph import StateGraph


@dataclass
class ActiveAgentCache:
    """Type-safe cache for active agents - NO dictionaries."""

    _entries: list[ActiveAgentEntry] = field(default_factory=list)

    def put(self, agent_id: UUID, state_graph: StateGraph) -> None:
        """Store active agent."""
        # Remove existing entry if present
        self.remove(agent_id)
        self._entries.append(ActiveAgentEntry(agent_id, state_graph))

    def get(self, agent_id: UUID) -> StateGraph | None:
        """Get state graph by agent ID."""
        for entry in self._entries:
            if entry.agent_id == agent_id:
                return entry.state_graph
        return None

    def remove(self, agent_id: UUID) -> StateGraph | None:
        """Remove and return state graph."""
        for i, entry in enumerate(self._entries):
            if entry.agent_id == agent_id:
                removed = self._entries.pop(i)
                return removed.state_graph
        return None

    def clear(self) -> None:
        """Clear all active agents."""
        self._entries.clear()

    def has_agent(self, agent_id: UUID) -> bool:
        """Check if agent is active."""
        return any(entry.agent_id == agent_id for entry in self._entries)

    def get_all_agent_ids(self) -> list[UUID]:
        """Get all active agent IDs."""
        return [entry.agent_id for entry in self._entries]

    def get_agent_count(self) -> int:
        """Get number of active agents."""
        return len(self._entries)
