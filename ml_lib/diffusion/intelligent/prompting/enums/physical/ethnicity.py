"""Ethnicity enum - replaces ethnicities from YAML."""

from ..base_prompt_enum import BasePromptEnum
from .hair_color import HairColor
from .hair_texture import HairTexture
from .eye_color import EyeColor
from .skin_tone import SkinTone


class Ethnicity(BasePromptEnum):
    """Ethnicity options for character generation.

    Values extracted from character_attributes.yaml.
    All valid ethnicities have equal probability (uniform distribution).

    Inherits prompt-friendly __str__() from BasePromptEnum.
    Each enum value provides metadata through properties.
    """

    CAUCASIAN = "caucasian"
    """Caucasian/European/White."""

    EAST_ASIAN = "east_asian"
    """East Asian (Chinese, Japanese, Korean, Mongolian)."""

    SOUTH_ASIAN = "south_asian"
    """South Asian (Indian, Pakistani, Bangladeshi, Sri Lankan, Nepalese)."""

    HISPANIC_LATINX = "hispanic_latinx"
    """Hispanic/Latinx (Mexican, Central/South American, Caribbean)."""

    AFRICAN_AMERICAN = "african_american"
    """African American/Black/Afro-American."""

    MIDDLE_EASTERN = "middle_eastern"
    """Middle Eastern (Arab, Persian, Turkish, Israeli)."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation.

        Examples:
            >>> Ethnicity.CAUCASIAN.keywords
            ('caucasian', 'european', 'white')
        """
        _keywords: dict[Ethnicity, tuple[str, ...]] = {
            Ethnicity.CAUCASIAN: ("caucasian", "european", "white"),
            Ethnicity.EAST_ASIAN: (
                "east asian",
                "chinese",
                "japanese",
                "korean",
                "mongolian",
            ),
            Ethnicity.SOUTH_ASIAN: (
                "south asian",
                "indian",
                "pakistan",
                "bangladesh",
                "sri lankan",
                "nepalese",
            ),
            Ethnicity.HISPANIC_LATINX: (
                "hispanic",
                "latinx",
                "mexican",
                "central american",
                "south american",
                "caribbean",
            ),
            Ethnicity.AFRICAN_AMERICAN: ("african american", "black", "afro-american"),
            Ethnicity.MIDDLE_EASTERN: (
                "middle eastern",
                "arab",
                "persian",
                "turkish",
                "israeli",
            ),
        }
        return _keywords[self]

    @property
    def prompt_weight(self) -> float:
        """Weight/emphasis for this attribute in prompts."""
        return 1.0  # All ethnicities have equal weight

    @property
    def hair_colors(self) -> tuple["HairColor", ...]:
        """Hair colors commonly associated with this ethnicity.

        Examples:
            >>> Ethnicity.EAST_ASIAN.hair_colors
            (HairColor.BLACK, HairColor.DARK_BROWN, HairColor.BROWN)
        """

        _hair_colors: dict[Ethnicity, tuple[HairColor, ...]] = {
            Ethnicity.CAUCASIAN: (
                HairColor.BLONDE,
                HairColor.BROWN,
                HairColor.BLACK,
                HairColor.RED,
                HairColor.GREY_SILVER,
                HairColor.WHITE,
            ),
            Ethnicity.EAST_ASIAN: (
                HairColor.BLACK,
                HairColor.DARK_BROWN,
                HairColor.BROWN,
            ),
            Ethnicity.SOUTH_ASIAN: (
                HairColor.BLACK,
                HairColor.DARK_BROWN,
                HairColor.BROWN,
            ),
            Ethnicity.HISPANIC_LATINX: (
                HairColor.BLACK,
                HairColor.DARK_BROWN,
                HairColor.BROWN,
                HairColor.BLONDE,
                HairColor.RED,
            ),
            Ethnicity.AFRICAN_AMERICAN: (
                HairColor.BLACK,
                HairColor.DARK_BROWN,
                HairColor.BROWN,
            ),
            Ethnicity.MIDDLE_EASTERN: (
                HairColor.BLACK,
                HairColor.DARK_BROWN,
                HairColor.BROWN,
                HairColor.BLONDE,
            ),
        }
        return _hair_colors[self]

    @property
    def hair_textures(self) -> tuple["HairTexture", ...]:
        """Hair textures commonly associated with this ethnicity."""

        _hair_textures: dict[Ethnicity, tuple[HairTexture, ...]] = {
            Ethnicity.CAUCASIAN: (
                HairTexture.STRAIGHT,
                HairTexture.WAVY,
                HairTexture.CURLY,
            ),
            Ethnicity.EAST_ASIAN: (HairTexture.STRAIGHT, HairTexture.WAVY),
            Ethnicity.SOUTH_ASIAN: (
                HairTexture.STRAIGHT,
                HairTexture.WAVY,
                HairTexture.CURLY,
            ),
            Ethnicity.HISPANIC_LATINX: (
                HairTexture.STRAIGHT,
                HairTexture.WAVY,
                HairTexture.CURLY,
            ),
            Ethnicity.AFRICAN_AMERICAN: (
                HairTexture.CURLY,
                HairTexture.COILY,
                HairTexture.TEXTURED,
            ),
            Ethnicity.MIDDLE_EASTERN: (
                HairTexture.WAVY,
                HairTexture.CURLY,
                HairTexture.STRAIGHT,
            ),
        }
        return _hair_textures[self]

    @property
    def eye_colors(self) -> tuple["EyeColor", ...]:
        """Eye colors commonly associated with this ethnicity."""

        _eye_colors: dict[Ethnicity, tuple[EyeColor, ...]] = {
            Ethnicity.CAUCASIAN: (
                EyeColor.BLUE,
                EyeColor.GREEN,
                EyeColor.HAZEL,
                EyeColor.BROWN,
                EyeColor.GRAY,
            ),
            Ethnicity.EAST_ASIAN: (EyeColor.BROWN, EyeColor.BLACK, EyeColor.HAZEL),
            Ethnicity.SOUTH_ASIAN: (EyeColor.BROWN, EyeColor.HAZEL, EyeColor.BLACK),
            Ethnicity.HISPANIC_LATINX: (
                EyeColor.BROWN,
                EyeColor.HAZEL,
                EyeColor.GREEN,
                EyeColor.BLACK,
            ),
            Ethnicity.AFRICAN_AMERICAN: (
                EyeColor.BROWN,
                EyeColor.BLACK,
                EyeColor.HAZEL,
            ),
            Ethnicity.MIDDLE_EASTERN: (
                EyeColor.BROWN,
                EyeColor.HAZEL,
                EyeColor.GREEN,
                EyeColor.BLACK,
            ),
        }
        return _eye_colors[self]

    @property
    def skin_tones(self) -> tuple["SkinTone", ...]:
        """Skin tones commonly associated with this ethnicity."""

        _skin_tones: dict[Ethnicity, tuple[SkinTone, ...]] = {
            Ethnicity.CAUCASIAN: (SkinTone.FAIR, SkinTone.LIGHT, SkinTone.MEDIUM),
            Ethnicity.EAST_ASIAN: (SkinTone.FAIR, SkinTone.LIGHT, SkinTone.MEDIUM),
            Ethnicity.SOUTH_ASIAN: (
                SkinTone.LIGHT,
                SkinTone.MEDIUM,
                SkinTone.MEDIUM_DARK,
            ),
            Ethnicity.HISPANIC_LATINX: (
                SkinTone.LIGHT,
                SkinTone.MEDIUM,
                SkinTone.MEDIUM_DARK,
            ),
            Ethnicity.AFRICAN_AMERICAN: (SkinTone.MEDIUM_DARK, SkinTone.DARK),
            Ethnicity.MIDDLE_EASTERN: (
                SkinTone.LIGHT,
                SkinTone.MEDIUM,
                SkinTone.MEDIUM_DARK,
            ),
        }
        return _skin_tones[self]
