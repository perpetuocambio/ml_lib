"""WebSocket message data for synthesis updates."""

from infrastructure.communication.http.websocket.types.geographic_marker_dto import (
    GeographicMarkerDto,
)
from infrastructure.communication.http.websocket.types.knowledge_edge_dto import (
    KnowledgeEdgeDto,
)
from infrastructure.communication.http.websocket.types.knowledge_node_dto import (
    KnowledgeNodeDto,
)
from infrastructure.communication.http.websocket.types.timeline_event_dto import (
    TimelineEventDto,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer
from pydantic import BaseModel


class SynthesisUpdateData(BaseModel):
    """Data payload for synthesis update WebSocket messages."""

    # Incremental data (newly added items)
    new_knowledge_nodes: list[KnowledgeNodeDto]
    new_knowledge_edges: list[KnowledgeEdgeDto]
    new_timeline_events: list[TimelineEventDto]
    new_geographic_markers: list[GeographicMarkerDto]

    # Summary statistics
    total_nodes: int
    total_edges: int
    total_events: int
    total_markers: int

    # Metadata
    synthesis_id: str
    generated_at_iso: str

    model_config = ProtocolSerializer.serialize_model_config(
        {"frozen": True}
    )  # Modern Pydantic config
