from dataclasses import dataclass

from ml_lib.diffusion.prompt.age.age_range_config import AgeRangeConfig
from ml_lib.diffusion.prompt.hair.hair_texture_config import HairTextureConfig
from ml_lib.diffusion.prompt.ethnic.ethnicity import EthnicityConfig
from ml_lib.diffusion.prompt.skin.prompting_config import SettingConfig
from ml_lib.diffusion.prompt.skin.skin_tone_config import SkinToneConfig


@dataclass(frozen=True)
class CharacterAttributesConfig:
    """Complete character attributes configuration."""

    skin_tones: dict[str, SkinToneConfig]
    ethnicities: dict[str, EthnicityConfig]
    eye_colors: dict[str, AttributeConfig]
    hair_colors: dict[str, AttributeConfig]
    hair_textures: dict[str, HairTextureConfig]
    body_types: dict[str, AttributeConfig]
    breast_sizes: dict[str, AttributeConfig]
    age_ranges: dict[str, AgeRangeConfig]
    settings: dict[str, SettingConfig]
    poses: dict[str, PoseConfig]
    clothing_conditions: dict[str, AttributeConfig]
    clothing_styles: dict[str, AttributeConfig]
    clothing_details: dict[str, AttributeConfig]
    accessories: dict[str, AttributeConfig]
    erotic_toys: dict[str, AttributeConfig]
    activities: dict[str, ActivityConfig]
    weather_conditions: dict[str, AttributeConfig]
    emotional_states: dict[str, AttributeConfig]
    environment_details: dict[str, AttributeConfig]
    artistic_styles: dict[str, ArtisticStyleConfig]
    physical_features: dict[str, AttributeConfig]
    body_sizes: dict[str, AttributeConfig]
    aesthetic_styles: dict[str, AttributeConfig]
    fantasy_races: dict[str, FantasyRaceConfig]
    special_effects: dict[str, SpecialEffectConfig]
    cosplay_styles: dict[str, CosplayStyleConfig]
