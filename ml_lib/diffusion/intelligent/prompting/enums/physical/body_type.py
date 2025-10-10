"""Body type enum - replaces body_types from YAML."""

from ..base_prompt_enum import BasePromptEnum


class BodyType(BasePromptEnum):
    """Body type options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid body types have equal probability (uniform distribution).
    """

    SLIM = "slim"
    """Slim/thin/petite/slender body."""

    ATHLETIC = "athletic"
    """Athletic/toned/fit/muscular body."""

    CURVY = "curvy"
    """Curvy/hourglass/curvaceous body."""

    FULL_FIGURED = "full_figured"
    """Full-figured/voluptuous/plus-size body."""

    MATURE = "mature"
    """Mature/older/aged body (age 50+)."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[BodyType, tuple[str, ...]] = {
            BodyType.SLIM: ("slim body", "thin body", "petite", "slender"),
            BodyType.ATHLETIC: ("athletic body", "toned body", "fit body", "muscular"),
            BodyType.CURVY: ("curvy body", "hourglass", "curvaceous", "rounded"),
            BodyType.FULL_FIGURED: ("full-figured", "voluptuous", "ample", "plus-size"),
            BodyType.MATURE: ("mature body", "older body", "aged body", "natural aging"),
        }
        return _keywords[self]

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        _min_ages: dict[BodyType, int] = {
            BodyType.SLIM: 18,
            BodyType.ATHLETIC: 18,
            BodyType.CURVY: 18,
            BodyType.FULL_FIGURED: 18,
            BodyType.MATURE: 50,
        }
        return _min_ages[self]

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80
