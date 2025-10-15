from enum import Enum


class AttributeType(Enum):
    """Types of character attributes."""

    # Core identity
    AGE_RANGE = "age_ranges"
    ETHNICITY = "ethnicities"
    SKIN_TONE = "skin_tones"

    # Physical features
    EYE_COLOR = "eye_colors"
    HAIR_COLOR = "hair_colors"
    HAIR_TEXTURE = "hair_textures"
    BODY_TYPE = "body_types"
    BREAST_SIZE = "breast_sizes"

    # Appearance details
    PHYSICAL_FEATURE = "physical_features"
    BODY_SIZE = "body_sizes"
    AESTHETIC_STYLE = "aesthetic_styles"
    FANTASY_RACE = "fantasy_races"

    # Clothing and accessories
    CLOTHING_STYLE = "clothing_styles"
    CLOTHING_CONDITION = "clothing_conditions"
    CLOTHING_DETAIL = "clothing_details"
    ACCESSORY = "accessories"

    # Activities and context
    ACTIVITY = "activities"
    EMOTIONAL_STATE = "emotional_states"
    POSE = "poses"
    ENVIRONMENT = "environments"
    SETTING = "settings"

    # Special attributes
    COSPLAY_STYLE = "cosplay_styles"
    ERATIC_TOY = "erotic_toys"
    WEATHER_CONDITION = "weather_conditions"
    SPECIAL_EFFECT = "special_effects"
    ARTISTIC_STYLE = "artistic_styles"
