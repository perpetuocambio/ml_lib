"""Proposal WebSocket message type."""

from infrastructure.communication.http.websocket.types.proposal_update_data import (
    ProposalUpdateData,
)
from pydantic import BaseModel


class ProposalWebSocketMessage(BaseModel):
    """WebSocket message for proposal updates."""

    type: str = "proposal_update"
    event_type: str
    project_id: str
    proposal_id: str
    data: ProposalUpdateData
