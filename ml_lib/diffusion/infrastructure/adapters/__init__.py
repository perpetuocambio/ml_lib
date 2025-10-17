"""Handler modules for diffusion operations."""

from ml_lib.diffusion.domain.services.attribute_collection import AttributeCollection
from ml_lib.diffusion.domain.services.random_selector import RandomAttributeSelector
from ml_lib.diffusion.infrastructure.memory_manager import MemoryManager
# CharacterGenerator removed - was dependent on deleted ConfigLoader
from ml_lib.diffusion.infrastructure.adapters.clip_vision_adapter import (
    CLIPVisionEncoder,
    CLIPVisionModelType,
    load_clip_vision,
)
from ml_lib.diffusion.infrastructure.adapters.ip_adapter import IPAdapterService
from ml_lib.diffusion.infrastructure.adapters.controlnet_adapter import ControlNetHandler
from ml_lib.diffusion.domain.services.adapter_registry import (
    AdapterRegistry,
    AdapterType,
    AdapterRegistration,
)

__all__ = [
    "AttributeCollection",
    "RandomAttributeSelector",
    "MemoryManager",
    # "CharacterGenerator",  # Removed - was dependent on deleted ConfigLoader
    "CLIPVisionEncoder",
    "CLIPVisionModelType",
    "load_clip_vision",
    "IPAdapterService",
    "ControlNetHandler",
    # Adapter registry
    "AdapterRegistry",
    "AdapterType",
    "AdapterRegistration",
]
