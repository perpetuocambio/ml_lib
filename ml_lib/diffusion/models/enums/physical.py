"""Physical character attributes enums - Consolidated from ml_lib/diffusion/intelligent/prompting/enums/physical/."""

from ml_lib.diffusion.models.enums.base import BasePromptEnum


class SkinTone(BasePromptEnum):
    """Skin tone options for character generation."""

    FAIR = "fair"
    LIGHT = "light"
    MEDIUM = "medium"
    MEDIUM_DARK = "medium_dark"
    DARK = "dark"

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        from .physical import Ethnicity

        return {
            SkinTone.FAIR: ("fair skin", "light skin", "pale skin"),
            SkinTone.LIGHT: ("light skin", "light-medium skin", "medium-fair skin"),
            SkinTone.MEDIUM: ("medium skin", "olive skin", "tan skin"),
            SkinTone.MEDIUM_DARK: (
                "medium-dark skin",
                "dark olive",
                "olive-brown skin",
            ),
            SkinTone.DARK: ("dark skin", "deep skin", "rich skin", "ebony skin"),
        }[self]

    @property
    def prompt_weight(self) -> float:
        """Weight/emphasis for this attribute in prompts."""
        return {
            SkinTone.FAIR: 1.1,
            SkinTone.LIGHT: 1.1,
            SkinTone.MEDIUM: 1.2,
            SkinTone.MEDIUM_DARK: 1.2,
            SkinTone.DARK: 1.2,
        }[self]

    @property
    def ethnicity_associations(self) -> tuple["Ethnicity", ...]:
        """Ethnicities commonly associated with this skin tone."""
        from .physical import Ethnicity

        return {
            SkinTone.FAIR: (Ethnicity.CAUCASIAN,),
            SkinTone.LIGHT: (Ethnicity.CAUCASIAN, Ethnicity.MIDDLE_EASTERN),
            SkinTone.MEDIUM: (
                Ethnicity.MIDDLE_EASTERN,
                Ethnicity.SOUTH_ASIAN,
                Ethnicity.HISPANIC_LATINX,
            ),
            SkinTone.MEDIUM_DARK: (
                Ethnicity.SOUTH_ASIAN,
                Ethnicity.MIDDLE_EASTERN,
                Ethnicity.AFRICAN_AMERICAN,
                Ethnicity.HISPANIC_LATINX,
            ),
            SkinTone.DARK: (Ethnicity.AFRICAN_AMERICAN,),
        }[self]

    @property
    def min_age(self) -> int:
        return 18

    @property
    def max_age(self) -> int:
        return 80

    def get_ethnicity_prompts(self) -> tuple[str, ...]:
        """Get all ethnicity associations as prompt-ready strings."""
        return tuple(str(eth) for eth in self.ethnicity_associations)


class EyeColor(BasePromptEnum):
    """Eye color options for character generation."""

    BROWN = "brown"
    BLACK = "black"
    BLUE = "blue"
    GREEN = "green"
    GRAY = "gray"
    HAZEL = "hazel"

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        return {
            EyeColor.BROWN: ("brown eyes", "dark brown eyes", "brown irises"),
            EyeColor.BLACK: ("black eyes", "dark eyes", "black irises"),
            EyeColor.BLUE: ("blue eyes", "sapphire eyes", "blue irises"),
            EyeColor.GREEN: ("green eyes", "emerald eyes", "green irises"),
            EyeColor.GRAY: ("gray eyes", "grey eyes", "steel eyes", "gray irises"),
            EyeColor.HAZEL: ("hazel eyes", "hazel irises"),
        }[self]

    @property
    def min_age(self) -> int:
        return 18

    @property
    def max_age(self) -> int:
        return 80


class HairTexture(BasePromptEnum):
    """Hair texture options for character generation."""

    STRAIGHT = "straight"
    WAVY = "wavy"
    CURLY = "curly"
    COILY = "coily"
    TEXTURED = "textured"

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        return {
            HairTexture.STRAIGHT: ("straight hair", "smooth hair", "sleek hair"),
            HairTexture.WAVY: ("wavy hair", "flowing hair", "loose waves"),
            HairTexture.CURLY: ("curly hair", "curled hair", "ringlets"),
            HairTexture.COILY: ("coily hair", "kinky hair", "tight coils"),
            HairTexture.TEXTURED: ("textured hair", "natural hair", "afro"),
        }[self]

    @property
    def prompt_weight(self) -> float:
        return 1.0


class HairColor(BasePromptEnum):
    """Hair color options for character generation."""

    BLACK = "black"
    DARK_BROWN = "dark_brown"
    BROWN = "brown"
    BLONDE = "blonde"
    RED = "red"
    GREY_SILVER = "grey_silver"
    WHITE = "white"

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        return {
            HairColor.BLACK: ("black hair", "raven hair", "ebony hair"),
            HairColor.DARK_BROWN: (
                "dark brown hair",
                "black-brown hair",
                "rich brown hair",
            ),
            HairColor.BROWN: (
                "brown hair",
                "light brown hair",
                "chestnut hair",
                "auburn",
            ),
            HairColor.BLONDE: (
                "blonde hair",
                "blond hair",
                "golden hair",
                "honey hair",
            ),
            HairColor.RED: (
                "red hair",
                "ginger hair",
                "auburn hair",
                "strawberry blonde",
            ),
            HairColor.GREY_SILVER: (
                "grey hair",
                "gray hair",
                "silver hair",
                "salt and pepper",
            ),
            HairColor.WHITE: ("white hair", "pure white hair", "snow white hair"),
        }[self]

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        return {
            HairColor.BLACK: 18,
            HairColor.DARK_BROWN: 18,
            HairColor.BROWN: 18,
            HairColor.BLONDE: 18,
            HairColor.RED: 18,
            HairColor.GREY_SILVER: 45,
            HairColor.WHITE: 60,
        }[self]

    @property
    def max_age(self) -> int:
        return 80


class Ethnicity(BasePromptEnum):
    """Ethnicity options for character generation."""

    CAUCASIAN = "caucasian"
    EAST_ASIAN = "east_asian"
    SOUTH_ASIAN = "south_asian"
    HISPANIC_LATINX = "hispanic_latinx"
    AFRICAN_AMERICAN = "african_american"
    MIDDLE_EASTERN = "middle_eastern"

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        return {
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
        }[self]

    @property
    def prompt_weight(self) -> float:
        return 1.0

    @property
    def hair_colors(self) -> tuple["HairColor", ...]:
        """Hair colors commonly associated with this ethnicity."""
        return {
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
        }[self]

    @property
    def hair_textures(self) -> tuple["HairTexture", ...]:
        """Hair textures commonly associated with this ethnicity."""
        return {
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
        }[self]

    @property
    def eye_colors(self) -> tuple["EyeColor", ...]:
        """Eye colors commonly associated with this ethnicity."""
        return {
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
        }[self]

    @property
    def skin_tones(self) -> tuple["SkinTone", ...]:
        """Skin tones commonly associated with this ethnicity."""
        return {
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
        }[self]


class PhysicalFeature(BasePromptEnum):
    """Physical feature options for character generation."""

    FRECKLES = "freckles"
    TATTOOS = "tattoos"
    PIERCINGS = "piercings"
    LARGE_FEATURES = "large_features"
    UNIQUE_FEATURES = "unique_features"

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        return {
            PhysicalFeature.FRECKLES: (
                "freckles",
                "freckled skin",
                "dotted skin",
                "light freckles",
                "dark freckles",
                "freckle pattern",
                "freckled face",
                "freckled shoulders",
            ),
            PhysicalFeature.TATTOOS: (
                "tattoos",
                "body tattoo",
                "sleeve tattoo",
                "back tattoo",
                "tattoo art",
                "inked skin",
                "tattooed",
                "tribal tattoo",
                "realistic tattoo",
            ),
            PhysicalFeature.PIERCINGS: (
                "piercings",
                "body piercing",
                "ear piercing",
                "nose piercing",
                "lip piercing",
                "navel piercing",
                "nipple piercing",
                "body jewelry",
                "pierced",
                "piercing jewelry",
            ),
            PhysicalFeature.LARGE_FEATURES: (
                "large breasts",
                "big breasts",
                "large nipples",
                "large labia",
                "pronounced features",
                "exaggerated features",
                "large areolas",
                "big areolas",
                "large body parts",
                "enhanced features",
            ),
            PhysicalFeature.UNIQUE_FEATURES: (
                "unique eyes",
                "red eyes",
                "colored eyes",
                "heterochromia",
                "distinctive features",
                "notable features",
                "remarkable features",
                "special features",
            ),
        }[self]

    @property
    def min_age(self) -> int:
        return 18

    @property
    def max_age(self) -> int:
        return 80


class BodyType(BasePromptEnum):
    """Body type options for character generation."""

    SLIM = "slim"
    ATHLETIC = "athletic"
    CURVY = "curvy"
    FULL_FIGURED = "full_figured"
    MATURE = "mature"

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        return {
            BodyType.SLIM: ("slim body", "thin body", "petite", "slender"),
            BodyType.ATHLETIC: ("athletic body", "toned body", "fit body", "muscular"),
            BodyType.CURVY: ("curvy body", "hourglass", "curvaceous", "rounded"),
            BodyType.FULL_FIGURED: ("full-figured", "voluptuous", "ample", "plus-size"),
            BodyType.MATURE: (
                "mature body",
                "older body",
                "aged body",
                "natural aging",
            ),
        }[self]

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        return {
            BodyType.SLIM: 18,
            BodyType.ATHLETIC: 18,
            BodyType.CURVY: 18,
            BodyType.FULL_FIGURED: 18,
            BodyType.MATURE: 50,
        }[self]

    @property
    def max_age(self) -> int:
        return 80


class BreastSize(BasePromptEnum):
    """Breast size options for character generation."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    EXTRA_LARGE = "extra_large"

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        return {
            BreastSize.SMALL: (
                "small breasts",
                "petite chest",
                "A cup",
                "modest chest",
            ),
            BreastSize.MEDIUM: (
                "medium breasts",
                "B cup",
                "average chest",
                "natural size",
            ),
            BreastSize.LARGE: ("large breasts", "C cup", "full chest", "voluptuous"),
            BreastSize.EXTRA_LARGE: (
                "extra large breasts",
                "D cup",
                "E cup",
                "ample chest",
            ),
        }[self]

    @property
    def min_age(self) -> int:
        return 18

    @property
    def max_age(self) -> int:
        return 80


class AgeRange(BasePromptEnum):
    """Age range options for character generation."""

    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    MILF = "milf"
    MATURE = "mature"

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        return {
            AgeRange.YOUNG_ADULT: ("young adult", "early twenties", "youthful"),
            AgeRange.ADULT: ("adult", "thirties", "mature adult"),
            AgeRange.MILF: ("milf", "mature woman", "older woman", "experienced"),
            AgeRange.MATURE: ("mature", "older", "senior"),
        }[self]


class BodySize(BasePromptEnum):
    """Body size options for character generation."""

    BBW = "bbw"
    SLIM = "slim"
    MUSCULAR = "muscular"
    PREGNANT = "pregnant"

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        return {
            BodySize.BBW: (
                "bbw",
                "curvy",
                "curvaceous",
                "curvy body",
                "voluptuous",
                "thick",
                "thick thighs",
                "big ass",
                "curvy figure",
                "ample curves",
            ),
            BodySize.SLIM: (
                "slim",
                "thin",
                "petite",
                "skinny",
                "slender",
                "willowy",
                "slim build",
                "petite build",
                "thin body",
                "narrow frame",
            ),
            BodySize.MUSCULAR: (
                "muscular",
                "toned",
                "fit",
                "athletic",
                "muscular body",
                "toned body",
                "fit body",
                "athletic build",
                "defined muscles",
                "well-toned",
            ),
            BodySize.PREGNANT: (
                "pregnant",
                "pregnant body",
                "pregnant belly",
                "expecting",
                "with child",
                "pregnant silhouette",
                "pregnant woman",
                "pregnant curves",
            ),
        }[self]

    @property
    def min_age(self) -> int:
        return 18

    @property
    def max_age(self) -> int:
        return 80


__all__ = [
    "SkinTone",
    "EyeColor",
    "HairTexture",
    "HairColor",
    "Ethnicity",
    "PhysicalFeature",
    "BodyType",
    "BreastSize",
    "AgeRange",
    "BodySize",
]
