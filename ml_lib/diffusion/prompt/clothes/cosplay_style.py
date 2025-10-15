from ml_lib.diffusion.prompt.common.base_prompt import BasePromptEnum


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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == CosplayStyle.ANIME:
            return [
                "anime character costume",
                "anime cosplay",
                "manga character",
                "japanese animation style",
                "otaku costume",
                "anime outfit",
                "manga outfit",
            ]
        elif self == CosplayStyle.CARTOON:
            return [
                "cartoon character costume",
                "cartoon cosplay",
                "animated character",
                "disney character",
                "cartoon outfit",
                "animated style",
            ]
        elif self == CosplayStyle.VIDEO_GAME:
            return [
                "video game character",
                "gaming cosplay",
                "game character costume",
                "gamer outfit",
                "game character",
                "video game costume",
            ]
        elif self == CosplayStyle.FANTASY:
            return [
                "fantasy character",
                "fantasy cosplay",
                "elf costume",
                "dwarf costume",
                "fantasy outfit",
                "magical character",
                "mythical character",
            ]
        elif self == CosplayStyle.SUPERHERO:
            return [
                "superhero costume",
                "comic book character",
                "superhero cosplay",
                "comic character",
                "superhero outfit",
                "cape and costume",
            ]
        elif self == CosplayStyle.HISTORICAL:
            return [
                "historical costume",
                "historical cosplay",
                "period costume",
                "historical outfit",
                "vintage costume",
                "period dress",
                "historical reenactment",
            ]
        elif self == CosplayStyle.MOVIE_TV:
            return [
                "movie character",
                "tv character",
                "film costume",
                "tv show character",
                "movie cosplay",
                "tv cosplay",
                "cinema character",
            ]
        elif self == CosplayStyle.ORIGINAL_CHARACTER:
            return [
                "original character",
                "oc costume",
                "custom costume",
                "original design",
                "unique character",
                "personal creation",
            ]
        else:
            return []

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        return 18

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80
