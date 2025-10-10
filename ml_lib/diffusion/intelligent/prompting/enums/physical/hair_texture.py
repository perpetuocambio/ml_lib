"""Hair texture enum - replaces hair_textures from YAML."""

from ..base_prompt_enum import BasePromptEnum


class HairTexture(BasePromptEnum):
    """Hair texture options for character generation.

    Values extracted from character_attributes.yaml.
    All valid hair textures have equal probability (uniform distribution).

    Each enum value provides metadata through properties.
    """

    STRAIGHT = "straight"
    """Straight/smooth/sleek hair."""

    WAVY = "wavy"
    """Wavy/slightly wavy hair."""

    CURLY = "curly"
    """Curly/tight curls hair."""

    COILY = "coily"
    """Coily/kinky/afro-textured hair."""

    TEXTURED = "textured"
    """Textured/natural texture hair."""
    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[HairTexture, tuple[str, ...]] = {
            HairTexture.STRAIGHT: ("straight hair", "smooth hair", "sleek hair"),
            HairTexture.WAVY: ("wavy hair", "flowing hair", "loose waves"),
            HairTexture.CURLY: ("curly hair", "curled hair", "ringlets"),
            HairTexture.COILY: ("coily hair", "kinky hair", "tight coils"),
            HairTexture.TEXTURED: ("textured hair", "natural hair", "afro"),
        }
        return _keywords[self]

    @property
    def prompt_weight(self) -> float:
        """Weight/emphasis for this attribute in prompts."""
        return 1.0

