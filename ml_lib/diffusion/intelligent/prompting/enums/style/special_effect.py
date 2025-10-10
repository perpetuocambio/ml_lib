"""Special effect enum - replaces special_effects from YAML."""

from ..base_prompt_enum import BasePromptEnum


class SpecialEffect(BasePromptEnum):
    """Special effect options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid special effects have equal probability (uniform distribution).
    """

    WET = "wet"
    """Wet/damp/sweaty skin."""

    CUM = "cum"
    """Cum/semen/body fluids."""

    STICKY = "sticky"
    """Sticky/slimy/coated skin."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[SpecialEffect, tuple[str, ...]] = {
            SpecialEffect.WET: ("wet", "wet skin", "damp", "moist", "wet look", "wet clothes", "sweaty", "drenched", "sopping wet"),
            SpecialEffect.CUM: ("cum", "semen", "bodily fluids", "cum on skin", "cum on face", "cum on body", "ejaculation", "body fluids"),
            SpecialEffect.STICKY: ("sticky", "sticky skin", "sticky body", "gooey", "slimy", "coated", "covered in substance", "oily", "greasy"),
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
