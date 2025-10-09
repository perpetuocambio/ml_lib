"""WebSocket connection manager."""

from uuid import UUID

from infrastructure.communication.http.websocket.types.agent_update_data import (
    AgentUpdateData,
)
from infrastructure.communication.http.websocket.types.agent_websocket_message import (
    AgentWebSocketMessage,
)
from infrastructure.communication.http.websocket.types.proposal_update_data import (
    ProposalUpdateData,
)
from infrastructure.communication.http.websocket.types.proposal_websocket_message import (
    ProposalWebSocketMessage,
)
from infrastructure.communication.http.websocket.types.synthesis_update_data import (
    SynthesisUpdateData,
)
from infrastructure.communication.http.websocket.types.synthesis_websocket_message import (
    SynthesisWebSocketMessage,
)
from infrastructure.communication.websocket.types.subscription_cache import (
    SubscriptionCache,
)
from websockets import WebSocketServerProtocol


class WebSocketManager:
    """Manages WebSocket connections and message broadcasting."""

    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: set[WebSocketServerProtocol] = set()
        self.project_subscriptions = SubscriptionCache()

    async def connect(self, websocket: WebSocketServerProtocol) -> None:
        """Accept a new WebSocket connection."""
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocketServerProtocol) -> None:
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)

        # Remove from project subscriptions
        self.project_subscriptions.remove_websocket_from_all_projects(websocket)

    async def subscribe_to_project(
        self, websocket: WebSocketServerProtocol, project_id: UUID
    ) -> None:
        """Subscribe connection to project updates."""
        self.project_subscriptions.add_subscription(project_id, websocket)

    def unsubscribe_from_project(
        self, websocket: WebSocketServerProtocol, project_id: UUID
    ) -> None:
        """Unsubscribe connection from project updates."""
        self.project_subscriptions.remove_subscription(project_id, websocket)

    async def send_personal_message(
        self, message: str, websocket: WebSocketServerProtocol
    ) -> None:
        """Send message to specific connection."""
        try:
            await websocket.send(message)
        except Exception:
            # Connection closed, remove it
            self.disconnect(websocket)

    async def broadcast_to_all(self, message: str) -> None:
        """Broadcast message to all connections."""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def broadcast_to_project(self, project_id: UUID, message: str) -> None:
        """Broadcast message to all connections subscribed to a project."""
        connections = self.project_subscriptions.get_project_websockets(project_id)
        if not connections:
            return
        disconnected = []

        for connection in connections:
            try:
                await connection.send(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def send_proposal_update(
        self,
        project_id: UUID,
        proposal_id: UUID,
        event_type: str,
        proposal_data: ProposalUpdateData,
    ) -> None:
        """Send proposal update to project subscribers."""
        message = ProposalWebSocketMessage(
            event_type=event_type,
            project_id=str(project_id),
            proposal_id=str(proposal_id),
            data=proposal_data,
        )

        message_json = message.model_dump_json()
        await self.broadcast_to_project(project_id, message_json)

    async def send_agent_update(
        self,
        project_id: UUID,
        agent_id: UUID,
        event_type: str,
        agent_data: AgentUpdateData,
    ) -> None:
        """Send agent update to project subscribers."""
        message = AgentWebSocketMessage(
            event_type=event_type,
            project_id=str(project_id),
            agent_id=str(agent_id),
            data=agent_data,
        )

        message_json = message.model_dump_json()
        await self.broadcast_to_project(project_id, message_json)

    def get_connection_count(self) -> int:
        """Get total number of active connections."""
        return len(self.active_connections)

    def get_project_subscriber_count(self, project_id: UUID) -> int:
        """Get number of subscribers for a specific project."""
        return len(self.project_subscriptions.get_project_websockets(project_id))

    async def send_synthesis_update(
        self,
        project_id: UUID,
        synthesis_id: str,
        synthesis_data: SynthesisUpdateData,
    ) -> None:
        """Send synthesis update to project subscribers."""
        message = SynthesisWebSocketMessage(
            event_type="synthesis_updated",
            project_id=str(project_id),
            synthesis_id=synthesis_id,
            data=synthesis_data,
        )

        message_json = message.model_dump_json()
        await self.broadcast_to_project(project_id, message_json)
