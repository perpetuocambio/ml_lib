"""Cosplay style enum - replaces cosplay_styles from YAML."""

from ..base_prompt_enum import BasePromptEnum


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
