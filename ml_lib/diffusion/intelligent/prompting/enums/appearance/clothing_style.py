"""Clothing style enum - replaces clothing_styles from YAML."""

from ..base_prompt_enum import BasePromptEnum


class ClothingStyle(BasePromptEnum):
    """Clothing style options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid clothing styles have equal probability (uniform distribution).
    """

    NUDE = "nude"
    """Nude/naked/completely nude."""

    LINGERIE = "lingerie"
    """Lingerie/underwear/sexy underwear."""

    CASUAL = "casual"
    """Casual wear/everyday clothes."""

    FORMAL = "formal"
    """Formal wear/evening dress/cocktail dress."""

    FETISH = "fetish"
    """Fetish wear/bondage gear/latex/leather."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[ClothingStyle, tuple[str, ...]] = {
            ClothingStyle.NUDE: ("nude", "naked", "completely nude", "fully nude", "nudity", "bare", "unclothed", "in the altogether", "in state of nature"),
            ClothingStyle.LINGERIE: ("lingerie", "underwear", "panties", "bra", "thong", "g-string", "bikini", "sexy lingerie", "seductive underwear"),
            ClothingStyle.CASUAL: ("casual wear", "everyday clothes", "t-shirt", "jeans", "casual outfit", "comfortable clothes", "daily wear"),
            ClothingStyle.FORMAL: ("formal wear", "evening dress", "cocktail dress", "gown", "suits", "formal attire", "elegant outfit"),
            ClothingStyle.FETISH: ("fetish wear", "bondage gear", "latex", "leather", "corset", "fishnet", "fetish outfit", "dominatrix outfit"),
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
