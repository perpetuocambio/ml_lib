"""Artistic style enum - replaces artistic_styles from YAML."""

from ..base_prompt_enum import BasePromptEnum


class ArtisticStyle(BasePromptEnum):
    """Artistic style options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid artistic styles have equal probability (uniform distribution).
    """

    PHOTOREALISTIC = "photorealistic"
    """Photorealistic/hyperrealistic/ultra realistic."""

    ANIME = "anime"
    """Anime/manga style."""

    CARTOON = "cartoon"
    """Cartoon/illustration/comic book style."""

    FANTASY = "fantasy"
    """Fantasy art/concept art."""

    VINTAGE = "vintage"
    """Vintage/retro/1950s-1970s style."""

    GOTHIC = "gothic"
    """Gothic/dark aesthetic."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[ArtisticStyle, tuple[str, ...]] = {
            ArtisticStyle.PHOTOREALISTIC: ("photorealistic", "hyperrealistic", "ultra realistic", "lifelike", "realistic", "photo", "photography", "professional photography", "film photography", "cinematic", "cinematic lighting", "studio photography"),
            ArtisticStyle.ANIME: ("anime style", "manga style", "japanese animation", "otaku art", "anime aesthetic", "chibi", "kawaii"),
            ArtisticStyle.CARTOON: ("cartoon style", "illustration", "comic book style", "hand-drawn", "animated", "disney style", "western cartoon", "comic art"),
            ArtisticStyle.FANTASY: ("fantasy art", "concept art", "fantasy illustration", "mythical art", "magical art style", "enchanted art", "fairy tale art"),
            ArtisticStyle.VINTAGE: ("vintage photo", "retro style", "vintage aesthetic", "1950s style", "1960s style", "1970s style", "vintage fashion", "retro photography", "old photo", "aged photo"),
            ArtisticStyle.GOTHIC: ("gothic style", "dark aesthetic", "goth", "dark fantasy", "macabre art", "dark romanticism", "victorian gothic"),
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
