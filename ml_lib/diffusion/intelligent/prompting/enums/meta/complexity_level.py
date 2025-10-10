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
