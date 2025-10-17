"""Models for the diffusion module."""

# Value objects
from ml_lib.diffusion.domain.value_objects_models.value_objects import (
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
    ProcessedPrompt,
)

# Core types
from ml_lib.diffusion.domain.value_objects_models.core import (
    AttributeType,
    AttributeDefinition,
)

# Enums
from ml_lib.diffusion.domain.value_objects_models.enums import (
    BasePromptEnum,
)

# Pipeline models
from ml_lib.diffusion.domain.value_objects_models.pipeline import (
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
from ml_lib.diffusion.domain.value_objects_models.memory import (
    SystemResources,
    OffloadConfig,
    OffloadStrategy,
    LoadedModel,
    EvictionPolicy,
)

# Prompt models
from ml_lib.diffusion.domain.value_objects_models.prompt import (
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

# Character models (moved to domain/entities/character.py)
# Import from there directly to avoid circular imports:
# from ml_lib.diffusion.domain.entities.character import GeneratedCharacter, ...

# LoRA models (LoRAInfo moved to pipeline.py, already imported above)

# Registry models
from ml_lib.diffusion.domain.value_objects_models.registry import (
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

# IP-Adapter models
from ml_lib.diffusion.domain.value_objects_models.ip_adapter import (
    IPAdapterVariant,
    IPAdapterConfig,
    ImageFeatures,
    ReferenceImage,
)

# ControlNet models
from ml_lib.diffusion.domain.value_objects_models.controlnet import (
    ControlType,
    ControlNetConfig,
    ControlImage,
    PreprocessorConfig,
)

# Content tags and NSFW classification
from ml_lib.diffusion.domain.value_objects_models.content_tags import (
    NSFWCategory,
    PromptTokenPriority,
    QualityTag,
    TokenClassification,
    PromptCompactionResult,
    DetectedActs,
    NSFWAnalysis,
    NSFWKeywordRegistry,
    NSFW_REGISTRY,
    classify_token,
    analyze_nsfw_content,
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
    "ProcessedPrompt",
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
    # Character (moved to domain/entities/character.py - import from there)
    # "GeneratedCharacter",
    # "SelectedAttributes",
    # "ValidationResult",
    # "CompatibilityMap",
    # "GenerationPreferences",
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
    # IP-Adapter
    "IPAdapterVariant",
    "IPAdapterConfig",
    "ImageFeatures",
    "ReferenceImage",
    # ControlNet
    "ControlType",
    "ControlNetConfig",
    "ControlImage",
    "PreprocessorConfig",
    # Content tags
    "NSFWCategory",
    "PromptTokenPriority",
    "QualityTag",
    "TokenClassification",
    "PromptCompactionResult",
    "DetectedActs",
    "NSFWAnalysis",
    "NSFWKeywordRegistry",
    "NSFW_REGISTRY",
    "classify_token",
    "analyze_nsfw_content",
]
