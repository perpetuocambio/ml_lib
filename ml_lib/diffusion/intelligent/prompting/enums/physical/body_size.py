"""Body size enum - replaces body_sizes from YAML."""

from ..base_prompt_enum import BasePromptEnum


class BodySize(BasePromptEnum):
    """Body size options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid body sizes have equal probability (uniform distribution).
    """

    BBW = "bbw"
    """BBW/curvy/thick."""

    SLIM = "slim"
    """Slim/thin/petite."""

    MUSCULAR = "muscular"
    """Muscular/toned/athletic."""

    PREGNANT = "pregnant"
    """Pregnant/expecting."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[BodySize, tuple[str, ...]] = {
            BodySize.BBW: ("bbw", "curvy", "curvaceous", "curvy body", "voluptuous", "thick", "thick thighs", "big ass", "curvy figure", "ample curves"),
            BodySize.SLIM: ("slim", "thin", "petite", "skinny", "slender", "willowy", "slim build", "petite build", "thin body", "narrow frame"),
            BodySize.MUSCULAR: ("muscular", "toned", "fit", "athletic", "muscular body", "toned body", "fit body", "athletic build", "defined muscles", "well-toned"),
            BodySize.PREGNANT: ("pregnant", "pregnant body", "pregnant belly", "expecting", "with child", "pregnant silhouette", "pregnant woman", "pregnant curves"),
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
