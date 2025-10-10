"""Breast size enum - replaces breast_sizes from YAML."""

from ..base_prompt_enum import BasePromptEnum


class BreastSize(BasePromptEnum):
    """Breast size options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid breast sizes have equal probability (uniform distribution).
    """

    SMALL = "small"
    """Small/petite/A cup."""

    MEDIUM = "medium"
    """Medium/B cup/average."""

    LARGE = "large"
    """Large/C cup/full."""

    EXTRA_LARGE = "extra_large"
    """Extra large/D-E cup/ample."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[BreastSize, tuple[str, ...]] = {
            BreastSize.SMALL: ("small breasts", "petite chest", "A cup", "modest chest"),
            BreastSize.MEDIUM: ("medium breasts", "B cup", "average chest", "natural size"),
            BreastSize.LARGE: ("large breasts", "C cup", "full chest", "voluptuous"),
            BreastSize.EXTRA_LARGE: ("extra large breasts", "D cup", "E cup", "ample chest"),
        }
        return _keywords[self]

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        return 18

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80
