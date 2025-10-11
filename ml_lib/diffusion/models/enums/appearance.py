"""Consolidated appearance enums for character generation.

This module consolidates all appearance-related enums from:
ml_lib/diffusion/intelligent/prompting/enums/appearance/

All enums are copied as-is without modifications.
"""

from enum import Enum


class BasePromptEnum(Enum):
    """Base class for all enums that need prompt-friendly string conversion.

    Provides automatic conversion of underscore-separated values to space-separated
    strings suitable for AI prompt generation.

    All enums in the prompting module should inherit from this base class for
    consistent string representation behavior.

    Examples:
        >>> class Color(BasePromptEnum):
        ...     DARK_BLUE = "dark_blue"
        >>> str(Color.DARK_BLUE)
        'dark blue'
        >>> f"Use {Color.DARK_BLUE} color"
        'Use dark blue color'
    """

    def __str__(self) -> str:
        """Get the prompt-friendly string representation.

        Automatically converts underscores to spaces for natural language output.

        Returns:
            Prompt-friendly string with underscores replaced by spaces.
        """
        return self.value.replace("_", " ")


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
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[ClothingDetail, tuple[str, ...]] = {
            ClothingDetail.EXPOSED: ("exposed", "revealing", "see-through", "transparent", "mesh", "fishnet", "sheer", "revealing outfit", "seductive", "provocative"),
            ClothingDetail.TIGHT: ("tight clothes", "tight dress", "tight panties", "tight bra", "form-fitting", "bodycon", "skin-tight", "clingy", "revealing fit", "fitted"),
            ClothingDetail.LOOSE: ("loose clothes", "baggy", "oversized", "loose fitting", "comfortable fit", "flowing", "drapey", "airy clothes"),
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
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[ClothingStyle, tuple[str, ...]] = {
            ClothingStyle.NUDE: ("nude", "naked", "completely nude", "fully nude", "nudity", "bare", "unclothed", "in the altogether", "in state of nature"),
            ClothingStyle.LINGERIE: ("lingerie", "underwear", "panties", "bra", "thong", "g-string", "bikini", "sexy lingerie", "seductive underwear"),
            ClothingStyle.CASUAL: ("casual wear", "everyday clothes", "t-shirt", "jeans", "casual outfit", "comfortable clothes", "daily wear"),
            ClothingStyle.FORMAL: ("formal wear", "evening dress", "cocktail dress", "gown", "suits", "formal attire", "elegant outfit"),
            ClothingStyle.FETISH: ("fetish wear", "bondage gear", "latex", "leather", "corset", "fishnet", "fetish outfit", "dominatrix outfit"),
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


class CosplayStyle(BasePromptEnum):
    """Cosplay style options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid cosplay styles have equal probability (uniform distribution).
    """

    ANIME = "anime"
    """Anime character costume."""

    CARTOON = "cartoon"
    """Cartoon character costume."""

    VIDEO_GAME = "video_game"
    """Video game character costume."""

    FANTASY = "fantasy"
    """Fantasy character costume."""

    SUPERHERO = "superhero"
    """Superhero/comic book character."""

    HISTORICAL = "historical"
    """Historical/period costume."""

    MOVIE_TV = "movie_tv"
    """Movie/TV character costume."""

    ORIGINAL_CHARACTER = "original_character"
    """Original character/OC costume."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[CosplayStyle, tuple[str, ...]] = {
            CosplayStyle.ANIME: ("anime character costume", "anime cosplay", "manga character", "japanese animation style", "otaku costume", "anime outfit", "manga outfit"),
            CosplayStyle.CARTOON: ("cartoon character costume", "cartoon cosplay", "animated character", "disney character", "cartoon outfit", "animated style"),
            CosplayStyle.VIDEO_GAME: ("video game character", "gaming cosplay", "game character costume", "gamer outfit", "game character", "video game costume"),
            CosplayStyle.FANTASY: ("fantasy character", "fantasy cosplay", "elf costume", "dwarf costume", "fantasy outfit", "magical character", "mythical character"),
            CosplayStyle.SUPERHERO: ("superhero costume", "comic book character", "superhero cosplay", "comic character", "superhero outfit", "cape and costume"),
            CosplayStyle.HISTORICAL: ("historical costume", "historical cosplay", "period costume", "historical outfit", "vintage costume", "period dress", "historical reenactment"),
            CosplayStyle.MOVIE_TV: ("movie character", "tv character", "film costume", "tv show character", "movie cosplay", "tv cosplay", "cinema character"),
            CosplayStyle.ORIGINAL_CHARACTER: ("original character", "oc costume", "custom costume", "original design", "unique character", "personal creation"),
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


__all__ = [
    "BasePromptEnum",
    "Accessory",
    "ClothingCondition",
    "ClothingDetail",
    "ClothingStyle",
    "CosplayStyle",
]
