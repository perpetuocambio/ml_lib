"""WebSocket message wrapper for synthesis updates."""

from infrastructure.communication.http.websocket.types.synthesis_update_data import (
    SynthesisUpdateData,
)
from infrastructure.serialization.protocol_serializer import ProtocolSerializer
from pydantic import BaseModel


class SynthesisWebSocketMessage(BaseModel):
    """WebSocket message for synthesis updates."""

    event_type: str  # "synthesis_updated"
    project_id: str
    synthesis_id: str
    data: SynthesisUpdateData

    model_config = ProtocolSerializer.serialize_model_config(
        {"frozen": True}
    )  # Modern Pydantic config
