"""Data structure for extracted entity persistence."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedEntityData:
    """Typed data structure for extracted entity database records."""

    entity_id: str
    name: str
    entity_type: str
    description: str
    confidence: float
    source_agent_id: str
    source_execution_id: str
    extraction_timestamp: str
    attributes_json: str | None
    related_entities_json: str | None
    geographic_info_json: str | None
    temporal_info_json: str | None
