"""Proposal update data for WebSocket messages."""

from infrastructure.communication.http.websocket.types.execution_result_data import (
    ExecutionResultData,
)
from infrastructure.communication.http.websocket.types.proposal_justification_data import (
    ProposalJustificationData,
)
from pydantic import BaseModel


class ProposalUpdateData(BaseModel):
    """Data for proposal updates."""

    agent_id: str
    proposal_type: str
    content: str
    justification: ProposalJustificationData
    status: str
    approved_by: str = ""
    execution_result: ExecutionResultData | None = None
    error_message: str = ""
