"""Clothing condition enum - replaces clothing_conditions from YAML."""

from ..base_prompt_enum import BasePromptEnum


class ClothingCondition(BasePromptEnum):
    """Clothing condition options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid clothing conditions have equal probability (uniform distribution).
    """

    INTACT = "intact"
    """Intact/perfect/pristine clothes."""

    TORN = "torn"
    """Torn/ripped clothes."""

    LOWERED = "lowered"
    """Lowered/pulled down clothes."""

    OPENED = "opened"
    """Opened/unbuttoned/unzipped clothes."""

    STAINED = "stained"
    """Stained/wet/dirty clothes."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[ClothingCondition, tuple[str, ...]] = {
            ClothingCondition.INTACT: ("intact clothes", "perfect clothes", "undamaged", "pristine condition", "well-maintained", "good condition", "tidy clothes"),
            ClothingCondition.TORN: ("torn clothes", "ripped clothes", "torn fabric", "torn dress", "ripped dress", "torn shirt", "torn outfit", "torn clothing", "torn panties", "ripped panties", "torn bra"),
            ClothingCondition.LOWERED: ("pulled down clothes", "lowered pants", "pulled down panties", "pants pulled down", "panties pulled down", "skirt lowered", "clothes lowered", "pulled to side", "pulled to one side", "bunched up", "clothes pulled down"),
            ClothingCondition.OPENED: ("unzipped", "unbuttoned", "open clothes", "unfastened", "undone", "open shirt", "open dress", "open blouse", "unbuckled", "undressed", "partially undressed", "open jacket", "open coat"),
            ClothingCondition.STAINED: ("stained clothes", "stained panties", "wet clothes", "wet panties", "cum stained", "cum on clothes", "cum on panties", "cum on dress", "soiled", "dirty clothes", "marked clothes", "sweaty clothes", "sweat stained", "bodily fluids", "urine", "wet look", "damp clothes"),
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
