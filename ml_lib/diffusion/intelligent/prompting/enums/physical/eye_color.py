"""Eye color enum - replaces eye_colors from YAML."""

from ..base_prompt_enum import BasePromptEnum


class EyeColor(BasePromptEnum):
    """Eye color options for character generation.

    Values extracted from character_attributes.yaml.
    All valid eye colors have equal probability (uniform distribution).

    Each enum value provides metadata through properties.
    """

    BROWN = "brown"
    """Brown eyes."""

    BLACK = "black"
    """Black/dark eyes."""

    BLUE = "blue"
    """Blue eyes."""

    GREEN = "green"
    """Green/emerald eyes."""

    GRAY = "gray"
    """Gray/grey/steel eyes."""

    HAZEL = "hazel"
    """Hazel/green-brown eyes."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation.

        Examples:
            >>> EyeColor.BROWN.keywords
            ('brown eyes', 'dark brown eyes', 'brown irises')
        """
        _keywords: dict[EyeColor, tuple[str, ...]] = {
            EyeColor.BROWN: ("brown eyes", "dark brown eyes", "brown irises"),
            EyeColor.BLACK: ("black eyes", "dark eyes", "black irises"),
            EyeColor.BLUE: ("blue eyes", "sapphire eyes", "blue irises"),
            EyeColor.GREEN: ("green eyes", "emerald eyes", "green irises"),
            EyeColor.GRAY: ("gray eyes", "grey eyes", "steel eyes", "gray irises"),
            EyeColor.HAZEL: ("hazel eyes", "hazel irises"),
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
