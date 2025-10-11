"""Models for the diffusion module."""

# Pipeline models
from ml_lib.diffusion.models.pipeline import (
    PipelineConfig,
    GenerationResult,
    GenerationMetadata,
    GenerationExplanation,
    Recommendations,
    BatchConfig,
)

# Memory models
from ml_lib.diffusion.models.memory import (
    SystemResources,
    OffloadConfig,
    LoadedModel,
)

# Prompt models
from ml_lib.diffusion.models.prompt import (
    PromptAnalysis,
    Intent,
    OptimizedParameters,
)

# Character models
from ml_lib.diffusion.models.character import (
    GeneratedCharacter,
    SelectedAttributes,
    ValidationResult,
    CompatibilityMap,
    GenerationPreferences,
)

# LoRA models
from ml_lib.diffusion.models.lora import (
    LoRARecommendation,
    LoRAInfo,
)

# Registry models
from ml_lib.diffusion.models.registry import (
    ModelMetadata,
    ModelFilter,
    DownloadResult,
)

__all__ = [
    # Pipeline
    "PipelineConfig",
    "GenerationResult",
    "GenerationMetadata",
    "GenerationExplanation",
    "Recommendations",
    "BatchConfig",
    # Memory
    "SystemResources",
    "OffloadConfig",
    "LoadedModel",
    # Prompt
    "PromptAnalysis",
    "Intent",
    "OptimizedParameters",
    # Character
    "GeneratedCharacter",
    "SelectedAttributes",
    "ValidationResult",
    "CompatibilityMap",
    "GenerationPreferences",
    # LoRA
    "LoRARecommendation",
    "LoRAInfo",
    # Registry
    "ModelMetadata",
    "ModelFilter",
    "DownloadResult",
]
