"""Age range enum - replaces age_ranges from YAML."""

from ..base_prompt_enum import BasePromptEnum


class AgeRange(BasePromptEnum):
    """Age range options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid age ranges have equal probability (uniform distribution).
    """

    YOUNG_ADULT = "young_adult"
    """Young adult (18-25 years)."""

    ADULT = "adult"
    """Adult (26-39 years)."""

    MILF = "milf"
    """MILF/mature woman (40-54 years)."""

    MATURE = "mature"
    """Mature/older/senior (55-80 years)."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[AgeRange, tuple[str, ...]] = {
            AgeRange.YOUNG_ADULT: ("young adult", "early twenties", "youthful"),
            AgeRange.ADULT: ("adult", "thirties", "mature adult"),
            AgeRange.MILF: ("milf", "mature woman", "older woman", "experienced"),
            AgeRange.MATURE: ("mature", "older", "senior"),
        }
        return _keywords[self]
