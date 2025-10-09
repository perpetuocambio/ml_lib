"""Proposal justification data for WebSocket messages."""

from pydantic import BaseModel


class ProposalJustificationData(BaseModel):
    """Type-safe container for proposal justification data."""

    reasoning: str = ""
    confidence_score: float = 0.0
    evidence_count: int = 0
    risk_assessment: str = ""
    method_used: str = ""
    data_quality: float = 0.0
    estimated_impact: int = 0
