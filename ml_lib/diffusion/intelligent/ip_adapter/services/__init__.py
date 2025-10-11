"""IP-Adapter services - Re-exports from handlers for backward compatibility."""

from ml_lib.diffusion.handlers.ip_adapter_handler import IPAdapterService
from ml_lib.diffusion.handlers.clip_vision_handler import (
    CLIPVisionEncoder,
    CLIPVisionModelType,
    load_clip_vision,
)

__all__ = [
    "IPAdapterService",
    "CLIPVisionEncoder",
    "CLIPVisionModelType",
    "load_clip_vision",
]
