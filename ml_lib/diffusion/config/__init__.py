"""Configuration for diffusion models.

This module provides:
- Path configuration for models (ModelPathConfig, ComfyUI paths)
- Central configuration system (DiffusionConfig)
- Configuration loading from files
"""

# Legacy path configuration
from ml_lib.diffusion.config.path_config import ModelPathConfig
from ml_lib.diffusion.config.comfyui_paths import (
    ComfyUIPathResolver,
    detect_comfyui_installation,
    create_comfyui_registry,
)

# New centralized configuration system
from ml_lib.diffusion.config.base import (
    DiffusionConfig,
    NegativePromptsConfig,
    LoRAConfig,
    ParameterRanges,
    VRAMConfig,
    VRAMPreset,
    DetailConfig,
    DetailPreset,
    SamplerConfig,
    ModelStrategies,
    GroupProfile,
    GroupProfiles,
    ConceptCategories,
)
from ml_lib.diffusion.config.types import (
    OptimizationLevel,
    SafetyLevel,
    SamplerName,
    ModelStrategy,
    TagList,
    WeightDict,
)
from ml_lib.diffusion.config.defaults import (
    get_default_config,
    set_default_config,
    reset_default_config,
)
from ml_lib.diffusion.config.loader import ConfigLoader

__all__ = [
    # Legacy path config
    "ModelPathConfig",
    "ComfyUIPathResolver",
    "detect_comfyui_installation",
    "create_comfyui_registry",
    # Main config
    "DiffusionConfig",
    # Sub-configs
    "NegativePromptsConfig",
    "LoRAConfig",
    "ParameterRanges",
    "VRAMConfig",
    "VRAMPreset",
    "DetailConfig",
    "DetailPreset",
    "SamplerConfig",
    "ModelStrategies",
    "GroupProfile",
    "GroupProfiles",
    "ConceptCategories",
    # Types
    "OptimizationLevel",
    "SafetyLevel",
    "SamplerName",
    "ModelStrategy",
    "TagList",
    "WeightDict",
    # Functions
    "get_default_config",
    "set_default_config",
    "reset_default_config",
    # Loader
    "ConfigLoader",
]
