"""Value objects for the diffusion module.

This module provides type-safe value objects to replace primitive types:
- Resolution: Replaces tuple[int, int] for image dimensions
- LoRAWeights: Replaces dict[str, float] for LoRA weights
- ConceptMap: Replaces dict[str, list[str]] for concept mappings
- EmphasisMap: Replaces dict[str, float] for keyword emphasis
- ReasoningMap: Replaces dict[str, str] for decision explanations
- Result objects: Replace tuple returns with structured data
"""

# Resolution
from .resolution import Resolution

# Weights
from .weights import (
    LoRAWeight,
    LoRAWeights,
    ParameterDelta,
    DeltaWeights,
    WeightConfig,
)

# Concepts
from .concepts import (
    Concept,
    ConceptMap,
    Emphasis,
    EmphasisMap,
)

# Reasoning
from .reasoning import (
    ReasoningEntry,
    ReasoningMap,
    LoRAReasoning,
    ParameterReasoning,
)

# Results
from .results import (
    SafetyStatus,
    SafetyCheckResult,
    PromptAnalysisResult,
    LoRARecommendationResult,
    ParameterOptimizationResult,
    NegativePromptResult,
)

# Generation Parameters
from .generation_params import (
    GenerationParameters,
    ParameterModification,
    ParameterModifications,
    FeedbackStatistics,
    TagCount,
)

# Pool Statistics
from .pool_stats import (
    PoolStatistics,
    ModelAccessInfo,
    ModelProtocol,
)

# Safety Results
from .safety_results import (
    PromptBlockResult,
    PromptSafetyResult,
)

# IP-Adapter Info
from .ip_adapter_info import (
    LoadedIPAdapterInfo,
    ModelRegistryProtocol,
    PipelineProtocol,
    CLIPVisionEncoderProtocol,
)

# Memory Stats
from .memory_stats import (
    MemoryStatistics,
    VAEProtocol,
    UNetProtocol,
    TransformerProtocol,
    ModelComponentProtocol,
)

# Parameter Modifications
from .parameter_modifications import (
    ParameterModificationEntry,
    ParameterModifications,
)

# Processed Prompt
from .processed_prompt import (
    ProcessedPrompt,
)

__all__ = [
    # Resolution
    "Resolution",
    # Weights
    "LoRAWeight",
    "LoRAWeights",
    "ParameterDelta",
    "DeltaWeights",
    "WeightConfig",
    # Concepts
    "Concept",
    "ConceptMap",
    "Emphasis",
    "EmphasisMap",
    # Reasoning
    "ReasoningEntry",
    "ReasoningMap",
    "LoRAReasoning",
    "ParameterReasoning",
    # Results
    "SafetyStatus",
    "SafetyCheckResult",
    "PromptAnalysisResult",
    "LoRARecommendationResult",
    "ParameterOptimizationResult",
    "NegativePromptResult",
    # Generation Parameters
    "GenerationParameters",
    "ParameterModification",
    "ParameterModifications",
    "FeedbackStatistics",
    "TagCount",
    # Pool Statistics
    "PoolStatistics",
    "ModelAccessInfo",
    "ModelProtocol",
    # Safety Results
    "PromptBlockResult",
    "PromptSafetyResult",
    # IP-Adapter Info
    "LoadedIPAdapterInfo",
    "ModelRegistryProtocol",
    "PipelineProtocol",
    "CLIPVisionEncoderProtocol",
    # Memory Stats
    "MemoryStatistics",
    "VAEProtocol",
    "UNetProtocol",
    "TransformerProtocol",
    "ModelComponentProtocol",
    # Parameter Modifications
    "ParameterModificationEntry",
    "ParameterModifications",
    # Processed Prompt
    "ProcessedPrompt",
]
