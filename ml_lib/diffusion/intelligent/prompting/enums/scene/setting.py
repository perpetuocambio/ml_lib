"""Setting enum - replaces settings from YAML."""

from ..base_prompt_enum import BasePromptEnum


class Setting(BasePromptEnum):
    """Setting/location options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid settings have equal probability (uniform distribution).
    """

    BEDROOM = "bedroom"
    """Bedroom/bed/intimate setting."""

    BATHROOM = "bathroom"
    """Bathroom/bathtub/shower."""

    LIVING_ROOM = "living_room"
    """Living room/couch/sofa."""

    OUTDOOR = "outdoor"
    """Outdoor/garden/terrace/patio."""

    STUDIO = "studio"
    """Photo studio/professional setting."""

    BEACH = "beach"
    """Beach/ocean/seashore."""

    PRIVATE_LUXURY = "private_luxury"
    """Luxury/upscale/high-end setting."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[Setting, tuple[str, ...]] = {
            Setting.BEDROOM: ("bedroom", "bed", "bed sheets", "intimate setting", "private room"),
            Setting.BATHROOM: ("bathroom", "bathtub", "shower", "mirror", "marble", "luxury bathroom"),
            Setting.LIVING_ROOM: ("living room", "couch", "sofa", "modern living", "cozy room"),
            Setting.OUTDOOR: ("outdoor", "garden", "terrace", "patio", "nature", "outdoor setting"),
            Setting.STUDIO: ("photo studio", "professional studio", "studio setting", "controlled lighting"),
            Setting.BEACH: ("beach", "ocean", "seashore", "coast", "sandy beach", "tropical setting"),
            Setting.PRIVATE_LUXURY: ("luxury setting", "upscale", "high-end", "elegant room", "premium location"),
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
