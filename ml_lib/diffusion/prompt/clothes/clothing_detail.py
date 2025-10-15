from ml_lib.diffusion.prompt.common.base_prompt import BasePromptEnum


class ClothingDetail(BasePromptEnum):
    """Clothing detail options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid clothing details have equal probability (uniform distribution).
    """

    EXPOSED = "exposed"
    """Exposed/revealing/see-through."""

    TIGHT = "tight"
    """Tight/form-fitting/bodycon."""

    LOOSE = "loose"
    """Loose/baggy/oversized."""

    @property
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == ClothingDetail.EXPOSED:
            return [
                "exposed",
                "revealing",
                "see-through",
                "transparent",
                "mesh",
                "fishnet",
                "sheer",
                "revealing outfit",
                "seductive",
                "provocative",
            ]
        elif self == ClothingDetail.TIGHT:
            return [
                "tight clothes",
                "tight dress",
                "tight panties",
                "tight bra",
                "form-fitting",
                "bodycon",
                "skin-tight",
                "clingy",
                "revealing fit",
                "fitted",
            ]
        elif self == ClothingDetail.LOOSE:
            return [
                "loose clothes",
                "baggy",
                "oversized",
                "loose fitting",
                "comfortable fit",
                "flowing",
                "drapey",
                "airy clothes",
            ]
        else:
            return []
