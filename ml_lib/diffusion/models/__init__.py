"""Models for the diffusion module."""

# Value objects
from ml_lib.diffusion.models.value_objects import (
    Resolution,
    LoRAWeight,
    LoRAWeights,
    ParameterDelta,
    DeltaWeights,
    WeightConfig,
    Concept,
    ConceptMap,
    Emphasis,
    EmphasisMap,
    ReasoningEntry,
    ReasoningMap,
    LoRAReasoning,
    ParameterReasoning,
    SafetyStatus,
    SafetyCheckResult,
    PromptAnalysisResult,
    LoRARecommendationResult,
    ParameterOptimizationResult,
    NegativePromptResult,
)

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
    LoRARecommendation,
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
    # Value objects
    "Resolution",
    "LoRAWeight",
    "LoRAWeights",
    "ParameterDelta",
    "DeltaWeights",
    "WeightConfig",
    "Concept",
    "ConceptMap",
    "Emphasis",
    "EmphasisMap",
    "ReasoningEntry",
    "ReasoningMap",
    "LoRAReasoning",
    "ParameterReasoning",
    "SafetyStatus",
    "SafetyCheckResult",
    "PromptAnalysisResult",
    "LoRARecommendationResult",
    "ParameterOptimizationResult",
    "NegativePromptResult",
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
    "LoRARecommendation",
    # Character
    "GeneratedCharacter",
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
