"""Aesthetic style enum - replaces aesthetic_styles from YAML."""

from ..base_prompt_enum import BasePromptEnum


class AestheticStyle(BasePromptEnum):
    """Aesthetic style options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid aesthetic styles have equal probability (uniform distribution).
    """

    GOTH = "goth"
    """Goth/gothic/dark aesthetic."""

    PUNK = "punk"
    """Punk/punk rock style."""

    NURSE = "nurse"
    """Nurse outfit/medical costume."""

    WITCH = "witch"
    """Witch/magical costume."""

    NUN = "nun"
    """Nun/religious costume."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[AestheticStyle, tuple[str, ...]] = {
            AestheticStyle.GOTH: ("goth", "gothic", "dark makeup", "dark clothing", "black clothing", "goth style", "dark aesthetic", "dark fashion", "black makeup"),
            AestheticStyle.PUNK: ("punk", "punk style", "punk fashion", "punk look", "leather jacket", "punk clothes", "punk makeup", "punk hair", "rebellious style"),
            AestheticStyle.NURSE: ("nurse", "nurse outfit", "medical costume", "nurse uniform", "hospital uniform", "nurse hat", "nurse shoes", "medical attire", "hospital costume"),
            AestheticStyle.WITCH: ("witch", "witch costume", "witch outfit", "witch hat", "witch aesthetic", "magical costume", "wizard", "sorceress", "spellcaster"),
            AestheticStyle.NUN: ("nun", "nun outfit", "religious costume", "nun habit", "convent clothing", "catholic costume", "religious attire", "ascetic clothing"),
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
