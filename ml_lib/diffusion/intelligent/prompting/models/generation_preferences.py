"""Generation preferences model - consolidates GenerationPreferences and CharacterGenerationContext."""

from dataclasses import dataclass

from ml_lib.diffusion.intelligent.prompting.enums import (
    SafetyLevel,
    CharacterFocus,
    QualityTarget,
)


@dataclass
class GenerationPreferences:
    """Preferences for character generation.

    This class consolidates the previously separate GenerationPreferences
    and CharacterGenerationContext classes.

    Uses enums instead of string literals for type safety.
    """

    # Targeting
    target_age: int | None = None
    """Specific target age for character (None = random)."""

    target_ethnicity: str | None = None
    """Specific target ethnicity (None = random)."""

    target_style: str | None = None
    """Specific target style (None = random)."""

    # Content control
    explicit_content_allowed: bool = True
    """Whether explicit/NSFW content is allowed."""

    safety_level: SafetyLevel = SafetyLevel.STRICT
    """Content filtering safety level."""

    # Visual preferences
    character_focus: CharacterFocus = CharacterFocus.PORTRAIT
    """Framing/composition focus for the character."""

    quality_target: QualityTarget = QualityTarget.HIGH
    """Target quality level for generation."""

    # Diversity
    diversity_target: float = 0.6
    """Target percentage of diverse/non-white characters (0.0-1.0)."""

    def __post_init__(self) -> None:
        """Validate preferences after initialization."""
        if self.diversity_target < 0.0 or self.diversity_target > 1.0:
            raise ValueError(f"diversity_target must be between 0 and 1, got {self.diversity_target}")

        if self.target_age is not None and (self.target_age < 18 or self.target_age > 100):
            raise ValueError(f"target_age must be between 18 and 100, got {self.target_age}")
