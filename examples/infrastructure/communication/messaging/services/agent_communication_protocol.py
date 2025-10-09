"""Protocol for agent-to-agent communication."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from uuid import UUID

from infrastructure.communication.messaging.types.agent_message import AgentMessage


class AgentCommunicationProtocol(ABC):
    """Protocol for agent-to-agent communication."""

    @abstractmethod
    async def send_message(
        self,
        sender_id: UUID,
        recipient_id: UUID,
        message: AgentMessage,
    ) -> bool:
        """Send message from one agent to another."""

    @abstractmethod
    async def receive_messages(
        self,
        agent_id: UUID,
    ) -> AsyncGenerator[AgentMessage, None]:
        """Stream incoming messages for an agent."""

    @abstractmethod
    async def get_active_agents(self, project_id: UUID) -> list[UUID]:
        """Get list of active agents in a project."""

    @abstractmethod
    async def subscribe_to_agent_channel(
        self,
        agent_id: UUID,
        listener_id: str,
    ) -> None:
        """Subscribe to an agent's communication channel."""

    @abstractmethod
    async def unsubscribe_from_agent_channel(
        self,
        agent_id: UUID,
        listener_id: str,
    ) -> None:
        """Unsubscribe from an agent's communication channel."""
