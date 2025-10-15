from ml_lib.diffusion.prompt.common.base_prompt import BasePromptEnum


class Accessory(BasePromptEnum):
    """Accessory options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid accessories have equal probability (uniform distribution).
    """

    JEWELRY = "jewelry"
    """Jewelry/necklace/earrings/bracelet."""

    HEADWEAR = "headwear"
    """Hat/cap/crown/headband."""

    EYEWEAR = "eyewear"
    """Glasses/sunglasses."""

    BAGS = "bags"
    """Handbag/purse/backpack."""

    FETISH_ACCESSORIES = "fetish_accessories"
    """Fetish accessories/leather gloves/collar."""

    @property
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == Accessory.JEWELRY:
            return [
                "jewelry",
                "necklace",
                "earrings",
                "bracelet",
                "rings",
                "jewels",
                "gold jewelry",
                "diamond jewelry",
                "silver accessories",
            ]
        elif self == Accessory.HEADWEAR:
            return [
                "hat",
                "cap",
                "crown",
                "tiara",
                "hair accessory",
                "headband",
                "hair clip",
                "headpiece",
                "hair decoration",
            ]
        elif self == Accessory.EYEWEAR:
            return [
                "glasses",
                "sunglasses",
                "eyeglasses",
                "designer glasses",
                "aviators",
                "frames",
                "specs",
            ]
        elif self == Accessory.BAGS:
            return [
                "handbag",
                "purse",
                "backpack",
                "clutch",
                "shoulder bag",
                "crossbody bag",
                "tote bag",
            ]
        elif self == Accessory.FETISH_ACCESSORIES:
            return [
                "fetish accessories",
                "leather gloves",
                "choker",
                "collar",
                "knee-high socks",
                "stockings",
                "garter belt",
                "feather boa",
            ]
        else:
            return []

    @property
    def min_age(self) -> int:
        return 18

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80
