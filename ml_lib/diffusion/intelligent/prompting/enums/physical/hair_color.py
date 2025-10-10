"""Hair color enum - replaces hair_colors from YAML."""

from ..base_prompt_enum import BasePromptEnum


class HairColor(BasePromptEnum):
    """Hair color options for character generation.

    Values extracted from character_attributes.yaml.
    All valid hair colors have equal probability (uniform distribution).

    Each enum value provides metadata through properties.
    """

    BLACK = "black"
    """Black/raven/ebony hair."""

    DARK_BROWN = "dark_brown"
    """Dark brown hair."""

    BROWN = "brown"
    """Brown/chestnut/auburn hair."""

    BLONDE = "blonde"
    """Blonde/golden/honey hair."""

    RED = "red"
    """Red/ginger/strawberry blonde hair."""

    GREY_SILVER = "grey_silver"
    """Grey/gray/silver hair (age 45+)."""

    WHITE = "white"
    """White/pure white hair (age 60+)."""
    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[HairColor, tuple[str, ...]] = {
            HairColor.BLACK: ("black hair", "raven hair", "ebony hair"),
            HairColor.DARK_BROWN: ("dark brown hair", "black-brown hair", "rich brown hair"),
            HairColor.BROWN: ("brown hair", "light brown hair", "chestnut hair", "auburn"),
            HairColor.BLONDE: ("blonde hair", "blond hair", "golden hair", "honey hair"),
            HairColor.RED: ("red hair", "ginger hair", "auburn hair", "strawberry blonde"),
            HairColor.GREY_SILVER: ("grey hair", "gray hair", "silver hair", "salt and pepper"),
            HairColor.WHITE: ("white hair", "pure white hair", "snow white hair"),
        }
        return _keywords[self]

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        _min_ages: dict[HairColor, int] = {
            HairColor.BLACK: 18,
            HairColor.DARK_BROWN: 18,
            HairColor.BROWN: 18,
            HairColor.BLONDE: 18,
            HairColor.RED: 18,
            HairColor.GREY_SILVER: 45,
            HairColor.WHITE: 60,
        }
        return _min_ages[self]

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80

