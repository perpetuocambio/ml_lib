"""WebSocket event type enumeration."""

from enum import Enum


class WebSocketEventType(Enum):
    """Types of WebSocket events for real-time communication."""

    # Proposal events
    PROPOSAL_CREATED = "created"
    PROPOSAL_APPROVED = "approved"
    PROPOSAL_REJECTED = "rejected"
    PROPOSAL_EXECUTED = "executed"
    PROPOSAL_FAILED = "failed"

    # Agent events
    AGENT_STARTED = "started"
    AGENT_STOPPED = "stopped"
    AGENT_STATUS_CHANGED = "status_changed"
    AGENT_MESSAGE = "message"
