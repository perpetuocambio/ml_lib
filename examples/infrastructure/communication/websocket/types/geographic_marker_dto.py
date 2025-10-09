"""DTO for geographic markers in WebSocket messages."""

from infrastructure.communication.http.websocket.types.synthesis_metadata import (
    SynthesisMetadata,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer
from pydantic import BaseModel


class GeographicMarkerDto(BaseModel):
    """DTO for geographic markers in WebSocket messages."""

    marker_id: str
    label: str
    marker_type: str
    latitude: float
    longitude: float
    description: str = ""
    address: str = ""
    confidence_score: float = 1.0
    metadata: SynthesisMetadata | None = None
    source_agent_id: str = ""
    created_at_iso: str = ""
    color: str = "#dc3545"
    icon: str = "marker"
    size: int = 25
    is_primary: bool = False
    cluster_group: str = ""
    related_node_ids: list[str] | None = None
    related_event_ids: list[str] | None = None

    model_config = ProtocolSerializer.serialize_model_config(
        {"frozen": True}
    )  # Modern Pydantic config
