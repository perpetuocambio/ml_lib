"""Physical feature enum - replaces physical_features from YAML."""

from ..base_prompt_enum import BasePromptEnum


class PhysicalFeature(BasePromptEnum):
    """Physical feature options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid physical features have equal probability (uniform distribution).
    """

    FRECKLES = "freckles"
    """Freckles/freckled skin."""

    TATTOOS = "tattoos"
    """Tattoos/body tattoo/inked skin."""

    PIERCINGS = "piercings"
    """Piercings/body piercing."""

    LARGE_FEATURES = "large_features"
    """Large features/pronounced features."""

    UNIQUE_FEATURES = "unique_features"
    """Unique eyes/heterochromia/distinctive features."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[PhysicalFeature, tuple[str, ...]] = {
            PhysicalFeature.FRECKLES: ("freckles", "freckled skin", "dotted skin", "light freckles", "dark freckles", "freckle pattern", "freckled face", "freckled shoulders"),
            PhysicalFeature.TATTOOS: ("tattoos", "body tattoo", "sleeve tattoo", "back tattoo", "tattoo art", "inked skin", "tattooed", "tribal tattoo", "realistic tattoo"),
            PhysicalFeature.PIERCINGS: ("piercings", "body piercing", "ear piercing", "nose piercing", "lip piercing", "navel piercing", "nipple piercing", "body jewelry", "pierced", "piercing jewelry"),
            PhysicalFeature.LARGE_FEATURES: ("large breasts", "big breasts", "large nipples", "large labia", "pronounced features", "exaggerated features", "large areolas", "big areolas", "large body parts", "enhanced features"),
            PhysicalFeature.UNIQUE_FEATURES: ("unique eyes", "red eyes", "colored eyes", "heterochromia", "distinctive features", "notable features", "remarkable features", "special features"),
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
