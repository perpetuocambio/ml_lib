"""WebSocket subscription cache - NO dictionaries."""

from dataclasses import dataclass, field
from uuid import UUID

from infrastructure.communication.websocket.types.subscription_entry import (
    SubscriptionEntry,
)
from websockets import WebSocketServerProtocol


@dataclass
class SubscriptionCache:
    """Type-safe cache for WebSocket subscriptions - NO dictionaries."""

    _subscriptions: list[SubscriptionEntry] = field(default_factory=list)

    def add_subscription(
        self, project_id: UUID, websocket: WebSocketServerProtocol
    ) -> None:
        """Add WebSocket subscription to project."""
        # Don't add if already exists
        if not self.has_subscription(project_id, websocket):
            self._subscriptions.append(SubscriptionEntry(project_id, websocket))

    def remove_subscription(
        self, project_id: UUID, websocket: WebSocketServerProtocol
    ) -> bool:
        """Remove WebSocket subscription from project."""
        for i, entry in enumerate(self._subscriptions):
            if entry.project_id == project_id and entry.websocket == websocket:
                self._subscriptions.pop(i)
                return True
        return False

    def get_project_websockets(self, project_id: UUID) -> list[WebSocketServerProtocol]:
        """Get all WebSockets subscribed to a project."""
        return [
            entry.websocket
            for entry in self._subscriptions
            if entry.project_id == project_id
        ]

    def remove_websocket_from_all_projects(
        self, websocket: WebSocketServerProtocol
    ) -> list[UUID]:
        """Remove WebSocket from all projects and return affected project IDs."""
        affected_projects = []
        remaining_subscriptions = []

        for entry in self._subscriptions:
            if entry.websocket == websocket:
                affected_projects.append(entry.project_id)
            else:
                remaining_subscriptions.append(entry)

        self._subscriptions[:] = remaining_subscriptions
        return list(set(affected_projects))  # Remove duplicates

    def has_subscription(
        self, project_id: UUID, websocket: WebSocketServerProtocol
    ) -> bool:
        """Check if WebSocket is subscribed to project."""
        return any(
            entry.project_id == project_id and entry.websocket == websocket
            for entry in self._subscriptions
        )

    def get_all_project_ids(self) -> list[UUID]:
        """Get all project IDs with subscriptions."""
        return list(set(entry.project_id for entry in self._subscriptions))

    def clear(self) -> None:
        """Clear all subscriptions."""
        self._subscriptions.clear()
