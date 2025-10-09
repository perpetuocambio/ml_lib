"""Agent proposal data transfer object for Infrastructure layer."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class AgentProposalData:
    """Data transfer object for agent proposal persistence."""

    proposal_id: str
    agent_id: str
    title: str
    description: str
    justification: str
    confidence: float
    priority: str
    status: str
    tool_name: str
    tool_arguments: str
    expected_outcome: str
    estimated_duration_minutes: int
    created_at: datetime
    approved_at: datetime | None
    approved_by_user_id: str | None
    rejection_reason: str | None
