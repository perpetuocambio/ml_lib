"""Message storage cache - NO dictionaries."""

from dataclasses import dataclass, field
from uuid import UUID

from infrastructure.agents.communication.types.message_storage_entry import (
    MessageStorageEntry,
)
from infrastructure.communication.messaging.types.agent_message import AgentMessage


@dataclass
class MessageStorageCache:
    """Type-safe cache for agent messages - NO dictionaries."""

    _entries: list[MessageStorageEntry] = field(default_factory=list)

    def add_message(self, agent_id: UUID, message: AgentMessage) -> None:
        """Add message to agent's storage."""
        # Find existing entry or create new one
        for entry in self._entries:
            if entry.agent_id == agent_id:
                entry.messages.append(message)
                return

        # Create new entry
        self._entries.append(MessageStorageEntry(agent_id, [message]))

    def get_messages(self, agent_id: UUID) -> list[AgentMessage]:
        """Get all messages for an agent."""
        for entry in self._entries:
            if entry.agent_id == agent_id:
                return entry.messages.copy()
        return []

    def clear_messages(self, agent_id: UUID) -> list[AgentMessage]:
        """Clear and return all messages for an agent."""
        for i, entry in enumerate(self._entries):
            if entry.agent_id == agent_id:
                messages = entry.messages.copy()
                self._entries.pop(i)
                return messages
        return []

    def has_messages(self, agent_id: UUID) -> bool:
        """Check if agent has messages."""
        for entry in self._entries:
            if entry.agent_id == agent_id:
                return len(entry.messages) > 0
        return False

    def get_message_count(self, agent_id: UUID) -> int:
        """Get number of messages for an agent."""
        for entry in self._entries:
            if entry.agent_id == agent_id:
                return len(entry.messages)
        return 0

    def get_all_agent_ids(self) -> list[UUID]:
        """Get all agent IDs with messages."""
        return [entry.agent_id for entry in self._entries]

    def clear(self) -> None:
        """Clear all message storage."""
        self._entries.clear()
