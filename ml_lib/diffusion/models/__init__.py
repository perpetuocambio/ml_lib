"""Models for the diffusion module."""

# Core types
from ml_lib.diffusion.models.core import (
    AttributeType,
    AttributeDefinition,
)

# Enums
from ml_lib.diffusion.models.enums import (
    BasePromptEnum,
)

# Pipeline models
from ml_lib.diffusion.models.pipeline import (
    PipelineConfig,
    GenerationResult,
    GenerationMetadata,
    GenerationExplanation,
    Recommendations,
    BatchConfig,
    OperationMode,
    GenerationConstraints,
    LoRAPreferences,
    MemorySettings,
    OllamaConfig,
    VariationStrategy,
    LoRASerializable,
    GenerationMetadataSerializable,
)

# Memory models
from ml_lib.diffusion.models.memory import (
    SystemResources,
    OffloadConfig,
    OffloadStrategy,
    LoadedModel,
    EvictionPolicy,
)

# Prompt models
from ml_lib.diffusion.models.prompt import (
    PromptAnalysis,
    ComplexityCategory,
    Intent,
    ArtisticStyle,
    ContentType,
    QualityLevel,
    OptimizedParameters,
    Priority,
    AttributeConfig,
    CharacterAttributeSet,
    GeneratedCharacter,
    LoRARecommendation,
)

# Character models
from ml_lib.diffusion.models.character import (
    SelectedAttributes,
    ValidationResult,
    CompatibilityMap,
    GenerationPreferences,
)

# LoRA models
from ml_lib.diffusion.models.lora import (
    LoRAInfo,
)

# Registry models
from ml_lib.diffusion.models.registry import (
    ModelMetadata,
    ModelFilter,
    DownloadResult,
    Source,
    ModelType,
    ModelFormat,
    BaseModel,
    SortBy,
    DownloadStatus,
)

__all__ = [
    # Core types
    "AttributeType",
    "AttributeDefinition",
    # Enums
    "BasePromptEnum",
    # Pipeline
    "PipelineConfig",
    "GenerationResult",
    "GenerationMetadata",
    "GenerationExplanation",
    "Recommendations",
    "BatchConfig",
    "OperationMode",
    "GenerationConstraints",
    "LoRAPreferences",
    "MemorySettings",
    "OllamaConfig",
    "VariationStrategy",
    "LoRASerializable",
    "GenerationMetadataSerializable",
    # Memory
    "SystemResources",
    "OffloadConfig",
    "OffloadStrategy",
    "LoadedModel",
    "EvictionPolicy",
    # Prompt
    "PromptAnalysis",
    "ComplexityCategory",
    "Intent",
    "ArtisticStyle",
    "ContentType",
    "QualityLevel",
    "OptimizedParameters",
    "Priority",
    "AttributeConfig",
    "CharacterAttributeSet",
    "GeneratedCharacter",
    "LoRARecommendation",
    # Character
    "SelectedAttributes",
    "ValidationResult",
    "CompatibilityMap",
    "GenerationPreferences",
    # LoRA
    "LoRAInfo",
    # Registry
    "ModelMetadata",
    "ModelFilter",
    "DownloadResult",
    "Source",
    "ModelType",
    "ModelFormat",
    "BaseModel",
    "SortBy",
    "DownloadStatus",
]
