"""Communication infrastructure module - handles external communication protocols."""

# HTTP communication
from infrastructure.communication.http import (
    HttpClient,
    HttpResponse,
    WebSocketServer,
)

# WebSocket communication
from infrastructure.communication.websocket import (
    AgentWebSocketMessage,
    ProposalWebSocketMessage,
    SynthesisWebSocketMessage,
    WebSocketEventType,
    WebSocketManager,
)

__all__ = [
    # HTTP
    "HttpClient",
    "HttpResponse",
    "WebSocketServer",
    # WebSocket
    "WebSocketManager",
    "WebSocketEventType",
    "AgentWebSocketMessage",
    "ProposalWebSocketMessage",
    "SynthesisWebSocketMessage",
]
