"""Handler modules for diffusion operations."""

from ml_lib.diffusion.handlers.attribute_collection import AttributeCollection
from ml_lib.diffusion.handlers.character_attribute_set import CharacterAttributeSet
from ml_lib.diffusion.handlers.config_loader import ConfigLoader
from ml_lib.diffusion.handlers.random_selector import RandomAttributeSelector
from ml_lib.diffusion.handlers.memory_manager import MemoryManager
from ml_lib.diffusion.handlers.character_generator import CharacterGenerator
from ml_lib.diffusion.handlers.clip_vision_handler import (
    CLIPVisionEncoder,
    CLIPVisionModelType,
    load_clip_vision,
)
from ml_lib.diffusion.handlers.ip_adapter_handler import IPAdapterService
from ml_lib.diffusion.handlers.controlnet_handler import ControlNetHandler
from ml_lib.diffusion.handlers.adapter_registry import (
    AdapterRegistry,
    AdapterType,
    AdapterRegistration,
)

__all__ = [
    "AttributeCollection",
    "CharacterAttributeSet",
    "ConfigLoader",
    "RandomAttributeSelector",
    "MemoryManager",
    "CharacterGenerator",
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
