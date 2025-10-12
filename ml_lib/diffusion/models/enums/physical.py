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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == SkinTone.FAIR:
            return ["fair skin", "light skin", "pale skin"]
        elif self == SkinTone.LIGHT:
            return ["light skin", "light-medium skin", "medium-fair skin"]
        elif self == SkinTone.MEDIUM:
            return ["medium skin", "olive skin", "tan skin"]
        elif self == SkinTone.MEDIUM_DARK:
            return [
                "medium-dark skin",
                "dark olive",
                "olive-brown skin",
            ]
        elif self == SkinTone.DARK:
            return ["dark skin", "deep skin", "rich skin", "ebony skin"]
        else:
            return []

    @property
    def prompt_weight(self) -> float:
        """Weight/emphasis for this attribute in prompts."""
        if self == SkinTone.FAIR:
            return 1.1
        elif self == SkinTone.LIGHT:
            return 1.1
        elif self == SkinTone.MEDIUM:
            return 1.2
        elif self == SkinTone.MEDIUM_DARK:
            return 1.2
        elif self == SkinTone.DARK:
            return 1.2
        else:
            return 0.0

    @property
    def ethnicity_associations(self) -> list["Ethnicity"]:
        """Ethnicities commonly associated with this skin tone."""
        if self == SkinTone.FAIR:
            return [Ethnicity.CAUCASIAN]
        elif self == SkinTone.LIGHT:
            return [Ethnicity.CAUCASIAN, Ethnicity.MIDDLE_EASTERN]
        elif self == SkinTone.MEDIUM:
            return [
                Ethnicity.MIDDLE_EASTERN,
                Ethnicity.SOUTH_ASIAN,
                Ethnicity.HISPANIC_LATINX,
            ]
        elif self == SkinTone.MEDIUM_DARK:
            return [
                Ethnicity.SOUTH_ASIAN,
                Ethnicity.MIDDLE_EASTERN,
                Ethnicity.AFRICAN_AMERICAN,
                Ethnicity.HISPANIC_LATINX,
            ]
        elif self == SkinTone.DARK:
            return [Ethnicity.AFRICAN_AMERICAN]
        else:
            return []

    @property
    def min_age(self) -> int:
        return 18

    @property
    def max_age(self) -> int:
        return 80

    def get_ethnicity_prompts(self) -> list[str]:
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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == EyeColor.BROWN:
            return ["brown eyes", "dark brown eyes", "brown irises"]
        elif self == EyeColor.BLACK:
            return ["black eyes", "dark eyes", "black irises"]
        elif self == EyeColor.BLUE:
            return ["blue eyes", "sapphire eyes", "blue irises"]
        elif self == EyeColor.GREEN:
            return ["green eyes", "emerald eyes", "green irises"]
        elif self == EyeColor.GRAY:
            return ["gray eyes", "grey eyes", "steel eyes", "gray irises"]
        elif self == EyeColor.HAZEL:
            return ["hazel eyes", "hazel irises"]
        else:
            return []

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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == HairTexture.STRAIGHT:
            return ["straight hair", "smooth hair", "sleek hair"]
        elif self == HairTexture.WAVY:
            return ["wavy hair", "flowing hair", "loose waves"]
        elif self == HairTexture.CURLY:
            return ["curly hair", "curled hair", "ringlets"]
        elif self == HairTexture.COILY:
            return ["coily hair", "kinky hair", "tight coils"]
        elif self == HairTexture.TEXTURED:
            return ["textured hair", "natural hair", "afro"]
        else:
            return []

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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == HairColor.BLACK:
            return ["black hair", "raven hair", "ebony hair"]
        elif self == HairColor.DARK_BROWN:
            return [
                "dark brown hair",
                "black-brown hair",
                "rich brown hair",
            ]
        elif self == HairColor.BROWN:
            return [
                "brown hair",
                "light brown hair",
                "chestnut hair",
                "auburn",
            ]
        elif self == HairColor.BLONDE:
            return [
                "blonde hair",
                "blond hair",
                "golden hair",
                "honey hair",
            ]
        elif self == HairColor.RED:
            return [
                "red hair",
                "ginger hair",
                "auburn hair",
                "strawberry blonde",
            ]
        elif self == HairColor.GREY_SILVER:
            return [
                "grey hair",
                "gray hair",
                "silver hair",
                "salt and pepper",
            ]
        elif self == HairColor.WHITE:
            return ["white hair", "pure white hair", "snow white hair"]
        else:
            return []

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        if self == HairColor.BLACK:
            return 18
        elif self == HairColor.DARK_BROWN:
            return 18
        elif self == HairColor.BROWN:
            return 18
        elif self == HairColor.BLONDE:
            return 18
        elif self == HairColor.RED:
            return 18
        elif self == HairColor.GREY_SILVER:
            return 45
        elif self == HairColor.WHITE:
            return 60
        else:
            return 18

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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == Ethnicity.CAUCASIAN:
            return ["caucasian", "european", "white"]
        elif self == Ethnicity.EAST_ASIAN:
            return [
                "east asian",
                "chinese",
                "japanese",
                "korean",
                "mongolian",
            ]
        elif self == Ethnicity.SOUTH_ASIAN:
            return [
                "south asian",
                "indian",
                "pakistan",
                "bangladesh",
                "sri lankan",
                "nepalese",
            ]
        elif self == Ethnicity.HISPANIC_LATINX:
            return [
                "hispanic",
                "latinx",
                "mexican",
                "central american",
                "south american",
                "caribbean",
            ]
        elif self == Ethnicity.AFRICAN_AMERICAN:
            return ["african american", "black", "afro-american"]
        elif self == Ethnicity.MIDDLE_EASTERN:
            return [
                "middle eastern",
                "arab",
                "persian",
                "turkish",
                "israeli",
            ]
        else:
            return []

    @property
    def prompt_weight(self) -> float:
        return 1.0

    @property
    def hair_colors(self) -> list["HairColor"]:
        """Hair colors commonly associated with this ethnicity."""
        if self == Ethnicity.CAUCASIAN:
            return [
                HairColor.BLONDE,
                HairColor.BROWN,
                HairColor.BLACK,
                HairColor.RED,
                HairColor.GREY_SILVER,
                HairColor.WHITE,
            ]
        elif self == Ethnicity.EAST_ASIAN:
            return [
                HairColor.BLACK,
                HairColor.DARK_BROWN,
                HairColor.BROWN,
            ]
        elif self == Ethnicity.SOUTH_ASIAN:
            return [
                HairColor.BLACK,
                HairColor.DARK_BROWN,
                HairColor.BROWN,
            ]
        elif self == Ethnicity.HISPANIC_LATINX:
            return [
                HairColor.BLACK,
                HairColor.DARK_BROWN,
                HairColor.BROWN,
                HairColor.BLONDE,
                HairColor.RED,
            ]
        elif self == Ethnicity.AFRICAN_AMERICAN:
            return [
                HairColor.BLACK,
                HairColor.DARK_BROWN,
                HairColor.BROWN,
            ]
        elif self == Ethnicity.MIDDLE_EASTERN:
            return [
                HairColor.BLACK,
                HairColor.DARK_BROWN,
                HairColor.BROWN,
                HairColor.BLONDE,
            ]
        else:
            return []

    @property
    def hair_textures(self) -> list["HairTexture"]:
        """Hair textures commonly associated with this ethnicity."""
        if self == Ethnicity.CAUCASIAN:
            return [
                HairTexture.STRAIGHT,
                HairTexture.WAVY,
                HairTexture.CURLY,
            ]
        elif self == Ethnicity.EAST_ASIAN:
            return [HairTexture.STRAIGHT, HairTexture.WAVY]
        elif self == Ethnicity.SOUTH_ASIAN:
            return [
                HairTexture.STRAIGHT,
                HairTexture.WAVY,
                HairTexture.CURLY,
            ]
        elif self == Ethnicity.HISPANIC_LATINX:
            return [
                HairTexture.STRAIGHT,
                HairTexture.WAVY,
                HairTexture.CURLY,
            ]
        elif self == Ethnicity.AFRICAN_AMERICAN:
            return [
                HairTexture.CURLY,
                HairTexture.COILY,
                HairTexture.TEXTURED,
            ]
        elif self == Ethnicity.MIDDLE_EASTERN:
            return [
                HairTexture.WAVY,
                HairTexture.CURLY,
                HairTexture.STRAIGHT,
            ]
        else:
            return []

    @property
    def eye_colors(self) -> list["EyeColor"]:
        """Eye colors commonly associated with this ethnicity."""
        if self == Ethnicity.CAUCASIAN:
            return [
                EyeColor.BLUE,
                EyeColor.GREEN,
                EyeColor.HAZEL,
                EyeColor.BROWN,
                EyeColor.GRAY,
            ]
        elif self == Ethnicity.EAST_ASIAN:
            return [EyeColor.BROWN, EyeColor.BLACK, EyeColor.HAZEL]
        elif self == Ethnicity.SOUTH_ASIAN:
            return [EyeColor.BROWN, EyeColor.HAZEL, EyeColor.BLACK]
        elif self == Ethnicity.HISPANIC_LATINX:
            return [
                EyeColor.BROWN,
                EyeColor.HAZEL,
                EyeColor.GREEN,
                EyeColor.BLACK,
            ]
        elif self == Ethnicity.AFRICAN_AMERICAN:
            return [
                EyeColor.BROWN,
                EyeColor.BLACK,
                EyeColor.HAZEL,
            ]
        elif self == Ethnicity.MIDDLE_EASTERN:
            return [
                EyeColor.BROWN,
                EyeColor.HAZEL,
                EyeColor.GREEN,
                EyeColor.BLACK,
            ]
        else:
            return []

    @property
    def skin_tones(self) -> list["SkinTone"]:
        """Skin tones commonly associated with this ethnicity."""
        if self == Ethnicity.CAUCASIAN:
            return [SkinTone.FAIR, SkinTone.LIGHT, SkinTone.MEDIUM]
        elif self == Ethnicity.EAST_ASIAN:
            return [SkinTone.FAIR, SkinTone.LIGHT, SkinTone.MEDIUM]
        elif self == Ethnicity.SOUTH_ASIAN:
            return [
                SkinTone.LIGHT,
                SkinTone.MEDIUM,
                SkinTone.MEDIUM_DARK,
            ]
        elif self == Ethnicity.HISPANIC_LATINX:
            return [
                SkinTone.LIGHT,
                SkinTone.MEDIUM,
                SkinTone.MEDIUM_DARK,
            ]
        elif self == Ethnicity.AFRICAN_AMERICAN:
            return [SkinTone.MEDIUM_DARK, SkinTone.DARK]
        elif self == Ethnicity.MIDDLE_EASTERN:
            return [
                SkinTone.LIGHT,
                SkinTone.MEDIUM,
                SkinTone.MEDIUM_DARK,
            ]
        else:
            return []


class PhysicalFeature(BasePromptEnum):
    """Physical feature options for character generation."""

    FRECKLES = "freckles"
    TATTOOS = "tattoos"
    PIERCINGS = "piercings"
    LARGE_FEATURES = "large_features"
    UNIQUE_FEATURES = "unique_features"

    @property
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == PhysicalFeature.FRECKLES:
            return [
                "freckles",
                "freckled skin",
                "dotted skin",
                "light freckles",
                "dark freckles",
                "freckle pattern",
                "freckled face",
                "freckled shoulders",
            ]
        elif self == PhysicalFeature.TATTOOS:
            return [
                "tattoos",
                "body tattoo",
                "sleeve tattoo",
                "back tattoo",
                "tattoo art",
                "inked skin",
                "tattooed",
                "tribal tattoo",
                "realistic tattoo",
            ]
        elif self == PhysicalFeature.PIERCINGS:
            return [
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
            ]
        elif self == PhysicalFeature.LARGE_FEATURES:
            return [
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
            ]
        elif self == PhysicalFeature.UNIQUE_FEATURES:
            return [
                "unique eyes",
                "red eyes",
                "colored eyes",
                "heterochromia",
                "distinctive features",
                "notable features",
                "remarkable features",
                "special features",
            ]
        else:
            return []

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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == BodyType.SLIM:
            return ["slim body", "thin body", "petite", "slender"]
        elif self == BodyType.ATHLETIC:
            return ["athletic body", "toned body", "fit body", "muscular"]
        elif self == BodyType.CURVY:
            return ["curvy body", "hourglass", "curvaceous", "rounded"]
        elif self == BodyType.FULL_FIGURED:
            return ["full-figured", "voluptuous", "ample", "plus-size"]
        elif self == BodyType.MATURE:
            return [
                "mature body",
                "older body",
                "aged body",
                "natural aging",
            ]
        else:
            return []

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        if self == BodyType.SLIM:
            return 18
        elif self == BodyType.ATHLETIC:
            return 18
        elif self == BodyType.CURVY:
            return 18
        elif self == BodyType.FULL_FIGURED:
            return 18
        elif self == BodyType.MATURE:
            return 50
        else:
            return 18

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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == BreastSize.SMALL:
            return [
                "small breasts",
                "petite chest",
                "A cup",
                "modest chest",
            ]
        elif self == BreastSize.MEDIUM:
            return [
                "medium breasts",
                "B cup",
                "average chest",
                "natural size",
            ]
        elif self == BreastSize.LARGE:
            return ["large breasts", "C cup", "full chest", "voluptuous"]
        elif self == BreastSize.EXTRA_LARGE:
            return [
                "extra large breasts",
                "D cup",
                "E cup",
                "ample chest",
            ]
        else:
            return []

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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == AgeRange.YOUNG_ADULT:
            return ["young adult", "early twenties", "youthful"]
        elif self == AgeRange.ADULT:
            return ["adult", "thirties", "mature adult"]
        elif self == AgeRange.MILF:
            return ["milf", "mature woman", "older woman", "experienced"]
        elif self == AgeRange.MATURE:
            return ["mature", "older", "senior"]
        else:
            return []


class BodySize(BasePromptEnum):
    """Body size options for character generation."""

    BBW = "bbw"
    SLIM = "slim"
    MUSCULAR = "muscular"
    PREGNANT = "pregnant"

    @property
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == BodySize.BBW:
            return [
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
            ]
        elif self == BodySize.SLIM:
            return [
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
            ]
        elif self == BodySize.MUSCULAR:
            return [
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
            ]
        elif self == BodySize.PREGNANT:
            return [
                "pregnant",
                "pregnant body",
                "pregnant belly",
                "expecting",
                "with child",
                "pregnant silhouette",
                "pregnant woman",
                "pregnant curves",
            ]
        else:
            return []

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
