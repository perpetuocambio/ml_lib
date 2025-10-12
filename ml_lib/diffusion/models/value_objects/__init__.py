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
]
