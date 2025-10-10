"""Quality target enum for generation quality level."""

from ..base_prompt_enum import BasePromptEnum


class QualityTarget(BasePromptEnum):
    """Target quality level for generation.

    Determines the overall quality and detail level of the generated output.
    """

    LOW = "low"
    """Low quality - faster generation, less detail."""

    MEDIUM = "medium"
    """Medium quality - balanced speed and detail."""

    HIGH = "high"
    """High quality - slower generation, high detail."""

    MASTERPIECE = "masterpiece"
    """Masterpiece quality - maximum detail and quality, slowest generation."""

    @property
    def description(self) -> str:
        """Get detailed description of this quality level."""
        _descriptions: dict[QualityTarget, str] = {
            QualityTarget.LOW: "Fast generation with basic quality suitable for testing and previews",
            QualityTarget.MEDIUM: "Balanced quality with good detail and reasonable generation time",
            QualityTarget.HIGH: "High-quality output with excellent detail and fine rendering",
            QualityTarget.MASTERPIECE: "Ultimate quality with maximum detail, refinement, and artistic excellence",
        }
        return _descriptions[self]

    @property
    def steps_multiplier(self) -> float:
        """Get generation steps multiplier (1.0 = baseline)."""
        _multipliers: dict[QualityTarget, float] = {
            QualityTarget.LOW: 0.5,  # 50% of baseline steps
            QualityTarget.MEDIUM: 1.0,  # Baseline
            QualityTarget.HIGH: 1.5,  # 150% of baseline
            QualityTarget.MASTERPIECE: 2.0,  # 200% of baseline
        }
        return _multipliers[self]

    @property
    def quality_score(self) -> int:
        """Get numeric quality score (1-10)."""
        _scores: dict[QualityTarget, int] = {
            QualityTarget.LOW: 3,
            QualityTarget.MEDIUM: 6,
            QualityTarget.HIGH: 8,
            QualityTarget.MASTERPIECE: 10,
        }
        return _scores[self]
