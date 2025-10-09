"""Agent update data for WebSocket messages."""

from infrastructure.communication.http.websocket.types.agent_metrics_data import (
    AgentMetricsData,
)
from pydantic import BaseModel


class AgentUpdateData(BaseModel):
    """Data for agent updates."""

    agent_type: str
    status: str
    message: str
    metrics: AgentMetricsData | None = None
