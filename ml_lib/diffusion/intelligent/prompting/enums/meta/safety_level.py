"""Safety level enum for content filtering."""

from ..base_prompt_enum import BasePromptEnum


class SafetyLevel(BasePromptEnum):
    """Safety level for content generation.

    Controls how strictly content is filtered for inappropriate material.
    """

    STRICT = "strict"
    """Strict filtering - blocks all potentially inappropriate content."""

    MODERATE = "moderate"
    """Moderate filtering - allows some mature content with restrictions."""

    RELAXED = "relaxed"
    """Relaxed filtering - minimal content restrictions."""

    @property
    def description(self) -> str:
        """Get detailed description of this safety level."""
        _descriptions: dict[SafetyLevel, str] = {
            SafetyLevel.STRICT: "Strict content filtering with maximum safety restrictions",
            SafetyLevel.MODERATE: "Balanced filtering allowing mature content with age verification",
            SafetyLevel.RELAXED: "Minimal filtering for adult audiences with content awareness",
        }
        return _descriptions[self]

    @property
    def filter_strength(self) -> int:
        """Get numeric filter strength (0-10, higher = stricter)."""
        _strengths: dict[SafetyLevel, int] = {
            SafetyLevel.STRICT: 10,
            SafetyLevel.MODERATE: 5,
            SafetyLevel.RELAXED: 2,
        }
        return _strengths[self]

    @property
    def blocks_explicit(self) -> bool:
        """Check if this level blocks explicit content."""
        _blocks: dict[SafetyLevel, bool] = {
            SafetyLevel.STRICT: True,
            SafetyLevel.MODERATE: False,
            SafetyLevel.RELAXED: False,
        }
        return _blocks[self]
