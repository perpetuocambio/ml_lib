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
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[Activity, tuple[str, ...]] = {
            Activity.INTIMATE: ("intimate position", "erotic pose", "sensual activity", "romantic setting", "intimate moment", "erotic scene"),
            Activity.SEXUAL: ("sex", "intercourse", "penetration", "oral", "anal", "position", "erotic act", "sexual activity", "lovemaking"),
            Activity.FOREPLAY: ("foreplay", "caressing", "touching", "kissing", "sensual massage", "erotic massage", "passionate kissing", "making out"),
            Activity.BDSM: ("bdsm", "domination", "submissive", "spanking", "impact play", "bondage scene", "role play"),
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
    def keywords(self) -> tuple[str, ...]:
        """Keywords used in prompt generation."""
        _keywords: dict[Pose, tuple[str, ...]] = {
            Pose.SITTING: ("sitting", "seated", "lounging", "reclining", "relaxed pose"),
            Pose.STANDING: ("standing", "erect", "upright", "posed", "straight posture"),
            Pose.KNEELING: ("kneeling", "kneeling position", "on knees", "bent position"),
            Pose.LYING: ("lying down", "reclining", "laying", "horizontal position"),
            Pose.INTIMATE_CLOSE: ("intimate pose", "close up", "erotic pose", "sensual position"),
            Pose.EXPLICIT_SEXUAL: ("sexual position", "sexual act", "intimate act", "erotic activity"),
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


__all__ = [
    "Setting",
    "Environment",
    "WeatherCondition",
    "Pose",
    "Activity",
]
