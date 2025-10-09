"""Agent-to-agent message entity."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from infrastructure.communication.messaging.types.agent_message_type import (
    AgentMessageType,
)


@dataclass(frozen=True)
class AgentMessage:
    """Message sent between agents."""

    message_id: UUID
    sender_agent_id: UUID
    recipient_agent_id: UUID
    message_type: AgentMessageType
    content: str
    context: str  # Additional context or metadata
    timestamp: datetime
    project_id: UUID
    requires_human_oversight: bool = False
