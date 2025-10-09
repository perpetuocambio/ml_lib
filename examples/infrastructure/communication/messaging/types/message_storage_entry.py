"""Single message storage entry."""

from dataclasses import dataclass
from uuid import UUID

from infrastructure.communication.messaging.types.agent_message import AgentMessage


@dataclass(frozen=True)
class MessageStorageEntry:
    """Single message storage entry."""

    agent_id: UUID
    messages: list[AgentMessage]
