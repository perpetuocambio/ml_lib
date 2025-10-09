"""Data structure for entity relationship persistence."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EntityRelationshipData:
    """Typed data structure for entity relationship database records."""

    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float
    evidence: str
    source_agent_id: str
    created_timestamp: str
