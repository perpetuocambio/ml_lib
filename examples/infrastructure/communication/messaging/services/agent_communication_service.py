"""Agent-to-agent communication service implementation."""

from collections.abc import AsyncGenerator
from uuid import UUID

from infrastructure.agents.communication.agent_communication_protocol import (
    AgentCommunicationProtocol,
)
from infrastructure.agents.communication.types.active_agents_cache import (
    ActiveAgentsCache,
)
from infrastructure.agents.communication.types.message_storage_cache import (
    MessageStorageCache,
)
from infrastructure.communication.http.websocket.services.websocket_manager import (
    WebSocketManager,
)
from infrastructure.communication.messaging.types.agent_message import AgentMessage


class AgentCommunicationService(AgentCommunicationProtocol):
    """Implementation of agent-to-agent communication."""

    def __init__(self, websocket_manager: WebSocketManager):
        """Initialize with WebSocket manager for real-time updates."""
        self._websocket_manager = websocket_manager
        self._message_storage = MessageStorageCache()
        self._active_agents = ActiveAgentsCache()

    async def send_message(
        self,
        sender_id: UUID,
        recipient_id: UUID,
        message: AgentMessage,
    ) -> bool:
        """Send message from one agent to another."""
        try:
            # Store message
            self._message_storage.add_message(recipient_id, message)

            # Send real-time notification via WebSocket
            await self._websocket_manager.send_to_channel(
                channel_id=f"agent_messages_{recipient_id}",
                message=self._serialize_message(message),
            )

            # Notify Director if oversight required
            if message.requires_human_oversight:
                await self._websocket_manager.send_to_channel(
                    channel_id=f"project_supervision_{message.project_id}",
                    message=self._serialize_supervision_alert(message),
                )

            return True

        except Exception:
            return False

    async def receive_messages(
        self,
        agent_id: UUID,
    ) -> AsyncGenerator[AgentMessage, None]:
        """Stream incoming messages for an agent."""
        messages = self._message_storage.get_messages(agent_id)

        for message in messages:
            yield message

        # Clear processed messages
        self._message_storage.clear_messages(agent_id)

    async def get_active_agents(self, project_id: UUID) -> list[UUID]:
        """Get list of active agents in a project."""
        return self._active_agents.get_agents(project_id)

    async def register_agent(self, project_id: UUID, agent_id: UUID) -> None:
        """Register an agent as active in a project."""
        self._active_agents.add_agent(project_id, agent_id)

    async def unregister_agent(self, project_id: UUID, agent_id: UUID) -> None:
        """Unregister an agent from a project."""
        self._active_agents.remove_agent(project_id, agent_id)

    async def subscribe_to_agent_channel(
        self,
        agent_id: UUID,
        listener_id: str,
    ) -> None:
        """Subscribe to an agent's communication channel."""
        # Implementation would depend on WebSocket manager capabilities
        pass

    async def unsubscribe_from_agent_channel(
        self,
        agent_id: UUID,
        listener_id: str,
    ) -> None:
        """Unsubscribe from an agent's communication channel."""
        # Implementation would depend on WebSocket manager capabilities
        pass

    def _serialize_message(self, message: AgentMessage) -> str:
        """Serialize message for WebSocket transmission."""
        return f'{{"type":"agent_message","sender":"{message.sender_agent_id}","recipient":"{message.recipient_agent_id}","message_type":"{message.message_type.value}","content":"{message.content}","timestamp":"{message.timestamp.isoformat()}"}}'

    def _serialize_supervision_alert(self, message: AgentMessage) -> str:
        """Serialize supervision alert for Director."""
        return f'{{"type":"supervision_required","message_id":"{message.message_id}","sender":"{message.sender_agent_id}","recipient":"{message.recipient_agent_id}","message_type":"{message.message_type.value}","requires_oversight":true}}'
