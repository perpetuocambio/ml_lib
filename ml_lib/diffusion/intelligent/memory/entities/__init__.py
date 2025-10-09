"""Entities for memory management."""

from ml_lib.diffusion.intelligent.memory.entities.system_resources import (
    SystemResources,
)
from ml_lib.diffusion.intelligent.memory.entities.offload_config import (
    OffloadConfig,
    OffloadStrategy,
)
from ml_lib.diffusion.intelligent.memory.entities.loaded_model import (
    LoadedModel,
    EvictionPolicy,
)

__all__ = [
    "SystemResources",
    "OffloadConfig",
    "OffloadStrategy",
    "LoadedModel",
    "EvictionPolicy",
]
