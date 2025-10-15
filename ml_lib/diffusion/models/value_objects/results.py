"""Result value objects for analysis and generation outputs.

This module provides type-safe result classes WITHOUT using dicts, tuples, or any.
"""

from dataclasses import dataclass, field
from enum import Enum

from ml_lib.diffusion.prompt.concepts import ConceptMap, EmphasisMap
from ml_lib.diffusion.models.value_objects.reasoning import ReasoningMap
from ml_lib.diffusion.models.value_objects.weights import LoRAWeights


@dataclass(frozen=True)
class LoRARecommendationResult:
    """Result of LoRA recommendation.

    Attributes:
        recommended_loras: Recommended LoRA weights.
        reasoning: Reasoning for recommendations.
        confidence: Overall confidence score (0.0 to 1.0).

    Example:
        >>> result = LoRARecommendationResult(
        ...     recommended_loras=lora_weights,
        ...     reasoning=reasoning_map,
        ...     confidence=0.85
        ... )
    """

    recommended_loras: LoRAWeights
    reasoning: ReasoningMap
    confidence: float

    def __post_init__(self) -> None:
        """Validate LoRA recommendation result."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    @property
    def lora_count(self) -> int:
        """Number of recommended LoRAs."""
        return self.recommended_loras.count

    @property
    def total_weight(self) -> float:
        """Total weight of all LoRAs."""
        return self.recommended_loras.total_weight

    @property
    def is_high_confidence(self) -> bool:
        """Check if confidence is high (>= 0.7)."""
        return self.confidence >= 0.7


@dataclass(frozen=True)
class ParameterOptimizationResult:
    """Result of parameter optimization.

    Attributes:
        steps: Optimized steps value.
        cfg_scale: Optimized CFG scale.
        sampler: Recommended sampler name.
        reasoning: Reasoning for parameter choices.

    Example:
        >>> result = ParameterOptimizationResult(
        ...     steps=35,
        ...     cfg_scale=9.0,
        ...     sampler="DPM++ 2M Karras",
        ...     reasoning=reasoning_map
        ... )
    """

    steps: int
    cfg_scale: float
    sampler: str
    reasoning: ReasoningMap

    def __post_init__(self) -> None:
        """Validate parameter optimization result."""
        if self.steps <= 0:
            raise ValueError(f"Steps must be positive, got {self.steps}")
        if self.cfg_scale <= 0:
            raise ValueError(f"CFG scale must be positive, got {self.cfg_scale}")
        if not self.sampler:
            raise ValueError("Sampler cannot be empty")


@dataclass(frozen=True)
class NegativePromptResult:
    """Result of negative prompt generation.

    Attributes:
        _negative_prompts: Internal list of negative prompt strings.
        reasoning: Reasoning for prompt selection.

    Example:
        >>> result = NegativePromptResult(
        ...     _negative_prompts=["low quality", "blurry"],
        ...     reasoning=reasoning_map
        ... )
    """

    _negative_prompts: list[str] = field(default_factory=list)
    reasoning: ReasoningMap = field(default_factory=lambda: ReasoningMap([]))

    def __post_init__(self) -> None:
        """Validate negative prompt result."""
        if not self._negative_prompts:
            raise ValueError("Negative prompts cannot be empty")

    @property
    def prompt_count(self) -> int:
        """Number of negative prompts."""
        return len(self._negative_prompts)

    def get_prompts(self) -> list[str]:
        """Get all negative prompts.

        Returns:
            List of negative prompt strings.
        """
        return self._negative_prompts.copy()

    def to_string(self, separator: str = ", ") -> str:
        """Convert to a single string.

        Args:
            separator: Separator between prompts.

        Returns:
            Concatenated negative prompt string.
        """
        return separator.join(self._negative_prompts)


__all__ = [
    "SafetyStatus",
    "SafetyCheckResult",
    "PromptAnalysisResult",
    "LoRARecommendationResult",
    "ParameterOptimizationResult",
    "NegativePromptResult",
]
