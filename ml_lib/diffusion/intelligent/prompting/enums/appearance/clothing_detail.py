"""Clothing detail enum - replaces clothing_details from YAML."""

from ..base_prompt_enum import BasePromptEnum


class ClothingDetail(BasePromptEnum):
    """Clothing detail options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid clothing details have equal probability (uniform distribution).
    """

    EXPOSED = "exposed"
    """Exposed/revealing/see-through."""

    TIGHT = "tight"
    """Tight/form-fitting/bodycon."""

    LOOSE = "loose"
    """Loose/baggy/oversized."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[ClothingDetail, tuple[str, ...]] = {
            ClothingDetail.EXPOSED: ("exposed", "revealing", "see-through", "transparent", "mesh", "fishnet", "sheer", "revealing outfit", "seductive", "provocative"),
            ClothingDetail.TIGHT: ("tight clothes", "tight dress", "tight panties", "tight bra", "form-fitting", "bodycon", "skin-tight", "clingy", "revealing fit", "fitted"),
            ClothingDetail.LOOSE: ("loose clothes", "baggy", "oversized", "loose fitting", "comfortable fit", "flowing", "drapey", "airy clothes"),
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
