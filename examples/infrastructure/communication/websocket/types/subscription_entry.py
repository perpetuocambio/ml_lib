"""Single subscription entry."""

from dataclasses import dataclass
from uuid import UUID

from websockets import WebSocketServerProtocol


@dataclass(frozen=True)
class SubscriptionEntry:
    """Single subscription entry."""

    project_id: UUID
    websocket: WebSocketServerProtocol
