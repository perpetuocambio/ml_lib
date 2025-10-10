"""Weather condition enum - replaces weather_conditions from YAML."""

from ..base_prompt_enum import BasePromptEnum


class WeatherCondition(BasePromptEnum):
    """Weather condition options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid weather conditions have equal probability (uniform distribution).
    """

    SUNNY = "sunny"
    """Sunny/bright day/clear sky."""

    CLOUDY = "cloudy"
    """Cloudy/overcast/gray sky."""

    RAINY = "rainy"
    """Rainy/rain/wet weather."""

    STORMY = "stormy"
    """Storm/thunder/lightning."""

    SNOWY = "snowy"
    """Snow/snowy/winter scene."""

    FOGGY = "foggy"
    """Fog/foggy/misty."""

    @property
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[WeatherCondition, tuple[str, ...]] = {
            WeatherCondition.SUNNY: ("sunny day", "bright day", "sunlight", "clear sky", "sunny weather", "bright sunlight"),
            WeatherCondition.CLOUDY: ("cloudy", "overcast", "cloudy sky", "gloomy", "cloud covered", "gray sky"),
            WeatherCondition.RAINY: ("rainy day", "rain", "raining", "wet weather", "rain drops", "stormy rain", "heavy rain"),
            WeatherCondition.STORMY: ("storm", "stormy weather", "thunder", "lightning", "stormy day", "thunder storm", "electric storm"),
            WeatherCondition.SNOWY: ("snow", "snowy", "snowy day", "snow covered", "winter scene", "snow flakes"),
            WeatherCondition.FOGGY: ("fog", "foggy", "misty", "foggy day", "mist", "hazy", "covered in fog"),
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
