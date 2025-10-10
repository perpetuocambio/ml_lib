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
