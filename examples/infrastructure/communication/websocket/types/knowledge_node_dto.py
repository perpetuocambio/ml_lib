"""DTO for knowledge graph nodes in WebSocket messages."""

from infrastructure.communication.http.websocket.types.synthesis_metadata import (
    SynthesisMetadata,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer
from pydantic import BaseModel


class KnowledgeNodeDto(BaseModel):
    """DTO for knowledge graph nodes in WebSocket messages."""

    model_config = ProtocolSerializer.serialize_model_config(
        {"frozen": True}
    )  # Modern Pydantic config

    node_id: str
    label: str
    node_type: str
    position_x: float
    position_y: float
    description: str
    confidence_score: float
    metadata: SynthesisMetadata
    source_agent_id: str
    created_at_iso: str
    color: str = "#1f77b4"
    size: int = 50
    is_primary: bool = False
