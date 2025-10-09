"""Agent WebSocket message type."""

from infrastructure.communication.http.websocket.types.agent_update_data import (
    AgentUpdateData,
)
from pydantic import BaseModel


class AgentWebSocketMessage(BaseModel):
    """WebSocket message for agent updates."""

    type: str = "agent_update"
    event_type: str
    project_id: str
    agent_id: str
    data: AgentUpdateData
