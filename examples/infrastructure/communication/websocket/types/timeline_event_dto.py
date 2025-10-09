"""DTO for timeline events in WebSocket messages."""

from infrastructure.communication.http.websocket.types.synthesis_metadata import (
    SynthesisMetadata,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer
from pydantic import BaseModel


class TimelineEventDto(BaseModel):
    """DTO for timeline events in WebSocket messages."""

    event_id: str
    title: str
    event_type: str
    timestamp_iso: str
    duration_minutes: int | None = None
    description: str = ""
    confidence_score: float = 1.0
    metadata: SynthesisMetadata | None = None
    source_agent_id: str = ""
    created_at_iso: str = ""
    color: str = "#28a745"
    icon: str = "circle"
    is_milestone: bool = False
    related_node_ids: list[str] | None = None

    model_config = ProtocolSerializer.serialize_model_config(
        {"frozen": True}
    )  # Modern Pydantic config
