"""Scene and environment enums - consolidated from ml_lib/diffusion/intelligent/prompting/enums/scene/."""

from ml_lib.diffusion.models.enums.base import BasePromptEnum


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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == WeatherCondition.SUNNY:
            return ["sunny day", "bright day", "sunlight", "clear sky", "sunny weather", "bright sunlight"]
        elif self == WeatherCondition.CLOUDY:
            return ["cloudy", "overcast", "cloudy sky", "gloomy", "cloud covered", "gray sky"]
        elif self == WeatherCondition.RAINY:
            return ["rainy day", "rain", "raining", "wet weather", "rain drops", "stormy rain", "heavy rain"]
        elif self == WeatherCondition.STORMY:
            return ["storm", "stormy weather", "thunder", "lightning", "stormy day", "thunder storm", "electric storm"]
        elif self == WeatherCondition.SNOWY:
            return ["snow", "snowy", "snowy day", "snow covered", "winter scene", "snow flakes"]
        elif self == WeatherCondition.FOGGY:
            return ["fog", "foggy", "misty", "foggy day", "mist", "hazy", "covered in fog"]
        else:
            raise ValueError(f"Unexpected WeatherCondition: {self}")

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        return 18

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80


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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == Environment.FOREST:
            return ["forest", "woods", "wooded area", "trees", "dense forest", "woodland", "timberland", "leafy environment", "natural forest"]
        elif self == Environment.INDOOR:
            return ["indoor", "inside", "interior", "room", "house", "apartment", "home", "indoor setting", "indoor location"]
        elif self == Environment.BEDROOM:
            return ["bedroom", "bed", "bedroom setting", "private room", "bedroom scene", "sleeping area", "bedroom environment"]
        elif self == Environment.OUTDOOR:
            return ["outdoor", "outside", "outdoor setting", "exterior", "open air", "outdoor location", "exterior location"]
        elif self == Environment.LUXURY:
            return ["luxury setting", "upscale environment", "luxurious", "expensive", "premium location", "high-end", "elegant setting"]
        elif self == Environment.NATURAL:
            return ["natural setting", "nature", "natural environment", "organic", "nature scene", "natural location", "outdoor nature"]
        else:
            raise ValueError(f"Unexpected Environment: {self}")

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        return 18

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80


class Activity(BasePromptEnum):
    """Activity options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid activities have equal probability (uniform distribution).
    """

    INTIMATE = "intimate"
    """Intimate/erotic/sensual activity (explicit)."""

    SEXUAL = "sexual"
    """Sexual/intercourse/penetration (explicit)."""

    FOREPLAY = "foreplay"
    """Foreplay/caressing/touching (explicit)."""

    BDSM = "bdsm"
    """BDSM/domination/spanking (explicit)."""

    @property
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == Activity.INTIMATE:
            return ["intimate position", "erotic pose", "sensual activity", "romantic setting", "intimate moment", "erotic scene"]
        elif self == Activity.SEXUAL:
            return ["sex", "intercourse", "penetration", "oral", "anal", "position", "erotic act", "sexual activity", "lovemaking"]
        elif self == Activity.FOREPLAY:
            return ["foreplay", "caressing", "touching", "kissing", "sensual massage", "erotic massage", "passionate kissing", "making out"]
        elif self == Activity.BDSM:
            return ["bdsm", "domination", "submissive", "spanking", "impact play", "bondage scene", "role play"]
        else:
            raise ValueError(f"Unexpected Activity: {self}")

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        return 18

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80


class Pose(BasePromptEnum):
    """Pose options for character generation.

    Values extracted from character_attributes.yaml.
    Each enum value provides metadata through properties.
    All valid poses have equal probability (uniform distribution).
    """

    SITTING = "sitting"
    """Sitting/seated/lounging pose."""

    STANDING = "standing"
    """Standing/upright/erect pose."""

    KNEELING = "kneeling"
    """Kneeling/on knees pose."""

    LYING = "lying"
    """Lying down/reclining pose."""

    INTIMATE_CLOSE = "intimate_close"
    """Intimate/close up/erotic pose (explicit)."""

    EXPLICIT_SEXUAL = "explicit_sexual"
    """Explicit sexual position/act (explicit)."""

    @property
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == Pose.SITTING:
            return ["sitting", "seated", "lounging", "reclining", "relaxed pose"]
        elif self == Pose.STANDING:
            return ["standing", "erect", "upright", "posed", "straight posture"]
        elif self == Pose.KNEELING:
            return ["kneeling", "kneeling position", "on knees", "bent position"]
        elif self == Pose.LYING:
            return ["lying down", "reclining", "laying", "horizontal position"]
        elif self == Pose.INTIMATE_CLOSE:
            return ["intimate pose", "close up", "erotic pose", "sensual position"]
        elif self == Pose.EXPLICIT_SEXUAL:
            return ["sexual position", "sexual act", "intimate act", "erotic activity"]
        else:
            raise ValueError(f"Unexpected Pose: {self}")

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        return 18

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80


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
    def keywords(self) -> list[str]:
        """Keywords used in prompt generation."""
        if self == Setting.BEDROOM:
            return ["bedroom", "bed", "bed sheets", "intimate setting", "private room"]
        elif self == Setting.BATHROOM:
            return ["bathroom", "bathtub", "shower", "mirror", "marble", "luxury bathroom"]
        elif self == Setting.LIVING_ROOM:
            return ["living room", "couch", "sofa", "modern living", "cozy room"]
        elif self == Setting.OUTDOOR:
            return ["outdoor", "garden", "terrace", "patio", "nature", "outdoor setting"]
        elif self == Setting.STUDIO:
            return ["photo studio", "professional studio", "studio setting", "controlled lighting"]
        elif self == Setting.BEACH:
            return ["beach", "ocean", "seashore", "coast", "sandy beach", "tropical setting"]
        elif self == Setting.PRIVATE_LUXURY:
            return ["luxury setting", "upscale", "high-end", "elegant room", "premium location"]
        else:
            raise ValueError(f"Unexpected Setting: {self}")

    @property
    def min_age(self) -> int:
        """Minimum age for this attribute."""
        return 18

    @property
    def max_age(self) -> int:
        """Maximum age for this attribute."""
        return 80


__all__ = [
    "Setting",
    "Environment",
    "WeatherCondition",
    "Pose",
    "Activity",
]
