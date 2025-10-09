"""Data structure for timeline event persistence."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TimelineEventData:
    """Typed data structure for timeline event database records."""

    event_id: str
    title: str
    description: str
    event_type: str
    timestamp: str
    confidence: float
    source_agent_id: str
    source_execution_id: str
    related_entities_json: str | None
    end_timestamp: str | None
    duration_description: str | None
    evidence: str | None
    significance_score: float | None
