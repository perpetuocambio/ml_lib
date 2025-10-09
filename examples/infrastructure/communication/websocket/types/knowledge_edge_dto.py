"""DTO for knowledge graph edges in WebSocket messages."""

from infrastructure.communication.http.websocket.types.synthesis_metadata import (
    SynthesisMetadata,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer
from pydantic import BaseModel


class KnowledgeEdgeDto(BaseModel):
    """DTO for knowledge graph edges in WebSocket messages."""

    model_config = ProtocolSerializer.serialize_model_config(
        {"frozen": True}
    )  # Modern Pydantic config

    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: str
    label: str
    description: str
    confidence_score: float
    metadata: SynthesisMetadata
    source_agent_id: str
    created_at_iso: str
    color: str = "#666666"
    width: int = 2
    is_directed: bool = True
    is_primary: bool = False
