"""Complexity level enum for attribute and generation complexity."""

from ..base_prompt_enum import BasePromptEnum


class ComplexityLevel(BasePromptEnum):
    """Complexity level for attributes and generation.

    Indicates the complexity and detail level of an attribute or generation task.
    """

    LOW = "low"
    """Low complexity - simple, straightforward."""

    MEDIUM = "medium"
    """Medium complexity - moderate detail and intricacy."""

    HIGH = "high"
    """High complexity - detailed, intricate, complex."""

    @property
    def description(self) -> str:
        """Get detailed description of this complexity level."""
        _descriptions: dict[ComplexityLevel, str] = {
            ComplexityLevel.LOW: "Simple and straightforward with minimal intricacy or special features",
            ComplexityLevel.MEDIUM: "Moderate complexity with balanced detail and some intricate elements",
            ComplexityLevel.HIGH: "Highly detailed and intricate with complex patterns and features",
        }
        return _descriptions[self]

    @property
    def complexity_score(self) -> int:
        """Get numeric complexity score (1-10)."""
        _scores: dict[ComplexityLevel, int] = {
            ComplexityLevel.LOW: 3,
            ComplexityLevel.MEDIUM: 6,
            ComplexityLevel.HIGH: 9,
        }
        return _scores[self]

    @property
    def token_weight(self) -> float:
        """Get prompt token weight multiplier for this complexity."""
        _weights: dict[ComplexityLevel, float] = {
            ComplexityLevel.LOW: 0.8,  # Less emphasis in prompt
            ComplexityLevel.MEDIUM: 1.0,  # Normal weight
            ComplexityLevel.HIGH: 1.3,  # Stronger emphasis
        }
        return _weights[self]
