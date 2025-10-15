from ml_lib.diffusion.prompt.common.base_prompt import BasePromptEnum


class ClothingStyle(BasePromptEnum):
    """Clothing style options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid clothing styles have equal probability (uniform distribution).
    """

    NUDE = "nude"
    """Nude/naked/completely nude."""

    LINGERIE = "lingerie"
    """Lingerie/underwear/sexy underwear."""

    CASUAL = "casual"
    """Casual wear/everyday clothes."""

    FORMAL = "formal"
    """Formal wear/evening dress/cocktail dress."""

    FETISH = "fetish"
    """Fetish wear/bondage gear/latex/leather."""

    @property
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == ClothingStyle.NUDE:
            return [
                "nude",
                "naked",
                "completely nude",
                "fully nude",
                "nudity",
                "bare",
                "unclothed",
                "in the altogether",
                "in state of nature",
            ]
        elif self == ClothingStyle.LINGERIE:
            return [
                "lingerie",
                "underwear",
                "panties",
                "bra",
                "thong",
                "g-string",
                "bikini",
                "sexy lingerie",
                "seductive underwear",
            ]
        elif self == ClothingStyle.CASUAL:
            return [
                "casual wear",
                "everyday clothes",
                "t-shirt",
                "jeans",
                "casual outfit",
                "comfortable clothes",
                "daily wear",
            ]
        elif self == ClothingStyle.FORMAL:
            return [
                "formal wear",
                "evening dress",
                "cocktail dress",
                "gown",
                "suits",
                "formal attire",
                "elegant outfit",
            ]
        elif self == ClothingStyle.FETISH:
            return [
                "fetish wear",
                "bondage gear",
                "latex",
                "leather",
                "corset",
                "fishnet",
                "fetish outfit",
                "dominatrix outfit",
            ]
        else:
            return []
