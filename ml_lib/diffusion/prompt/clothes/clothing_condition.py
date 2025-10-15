from ml_lib.diffusion.prompt.common.base_prompt import BasePromptEnum


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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == ClothingCondition.INTACT:
            return [
                "intact clothes",
                "perfect clothes",
                "undamaged",
                "pristine condition",
                "well-maintained",
                "good condition",
                "tidy clothes",
            ]
        elif self == ClothingCondition.TORN:
            return [
                "torn clothes",
                "ripped clothes",
                "torn fabric",
                "torn dress",
                "ripped dress",
                "torn shirt",
                "torn outfit",
                "torn clothing",
                "torn panties",
                "ripped panties",
                "torn bra",
            ]
        elif self == ClothingCondition.LOWERED:
            return [
                "pulled down clothes",
                "lowered pants",
                "pulled down panties",
                "pants pulled down",
                "panties pulled down",
                "skirt lowered",
                "clothes lowered",
                "pulled to side",
                "pulled to one side",
                "bunched up",
                "clothes pulled down",
            ]
        elif self == ClothingCondition.OPENED:
            return [
                "unzipped",
                "unbuttoned",
                "open clothes",
                "unfastened",
                "undone",
                "open shirt",
                "open dress",
                "open blouse",
                "unbuckled",
                "undressed",
                "partially undressed",
                "open jacket",
                "open coat",
            ]
        elif self == ClothingCondition.STAINED:
            return [
                "stained clothes",
                "stained panties",
                "wet clothes",
                "wet panties",
                "cum stained",
                "cum on clothes",
                "cum on panties",
                "cum on dress",
                "soiled",
                "dirty clothes",
                "marked clothes",
                "sweaty clothes",
                "sweat stained",
                "bodily fluids",
                "urine",
                "wet look",
                "damp clothes",
            ]
        else:
            return []
