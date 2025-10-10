"""Environment enum - replaces environment_details from YAML."""

from ..base_prompt_enum import BasePromptEnum


class Environment(BasePromptEnum):
    """Environment detail options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid environments have equal probability (uniform distribution).
    """

    FOREST = "forest"
    """Forest/woods/woodland."""

    INDOOR = "indoor"
    """Indoor/inside/interior."""

    BEDROOM = "bedroom"
    """Bedroom/bed setting."""

    OUTDOOR = "outdoor"
    """Outdoor/outside/exterior."""

    LUXURY = "luxury"
    """Luxury/upscale/high-end setting."""

    NATURAL = "natural"
    """Natural/nature setting."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[Environment, tuple[str, ...]] = {
            Environment.FOREST: ("forest", "woods", "wooded area", "trees", "dense forest", "woodland", "timberland", "leafy environment", "natural forest"),
            Environment.INDOOR: ("indoor", "inside", "interior", "room", "house", "apartment", "home", "indoor setting", "indoor location"),
            Environment.BEDROOM: ("bedroom", "bed", "bedroom setting", "private room", "bedroom scene", "sleeping area", "bedroom environment"),
            Environment.OUTDOOR: ("outdoor", "outside", "outdoor setting", "exterior", "open air", "outdoor location", "exterior location"),
            Environment.LUXURY: ("luxury setting", "upscale environment", "luxurious", "expensive", "premium location", "high-end", "elegant setting"),
            Environment.NATURAL: ("natural setting", "nature", "natural environment", "organic", "nature scene", "natural location", "outdoor nature"),
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
