"""Accessory enum - replaces accessories from YAML."""

from ..base_prompt_enum import BasePromptEnum


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
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[Accessory, tuple[str, ...]] = {
            Accessory.JEWELRY: ("jewelry", "necklace", "earrings", "bracelet", "rings", "jewels", "gold jewelry", "diamond jewelry", "silver accessories"),
            Accessory.HEADWEAR: ("hat", "cap", "crown", "tiara", "hair accessory", "headband", "hair clip", "headpiece", "hair decoration"),
            Accessory.EYEWEAR: ("glasses", "sunglasses", "eyeglasses", "designer glasses", "aviators", "frames", "specs"),
            Accessory.BAGS: ("handbag", "purse", "backpack", "clutch", "shoulder bag", "crossbody bag", "tote bag"),
            Accessory.FETISH_ACCESSORIES: ("fetish accessories", "leather gloves", "choker", "collar", "knee-high socks", "stockings", "garter belt", "feather boa"),
        }
        return _keywords[self]

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        _min_ages: dict[Accessory, int] = {
            Accessory.JEWELRY: 16,
            Accessory.HEADWEAR: 16,
            Accessory.EYEWEAR: 16,
            Accessory.BAGS: 16,
            Accessory.FETISH_ACCESSORIES: 18,
        }
        return _min_ages[self]

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80
