"""Prompting configuration using dataclasses - replaces YAML configuration files.

This module converts all YAML configuration files to type-safe dataclasses:
- character_attributes.yaml
- concept_categories.yaml
- lora_filters.yaml
- prompting_strategies.yaml
- generation_profiles.yaml

All configuration is now in pure Python with no YAML dependencies.
"""

from dataclasses import dataclass, field
from typing import Literal


# ============================================================================
# CHARACTER ATTRIBUTES CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class AttributeConfig:
    """Base configuration for character attributes."""
    keywords: tuple[str, ...]
    probability: float = 1.0
    min_age: int = 18
    max_age: int = 80


@dataclass(frozen=True)
class SkinToneConfig(AttributeConfig):
    """Configuration for skin tone attributes."""
    prompt_weight: float = 1.1
    ethnicity_associations: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class EthnicityConfig(AttributeConfig):
    """Configuration for ethnicity attributes."""
    prompt_weight: float = 1.0
    hair_colors: tuple[str, ...] = field(default_factory=tuple)
    hair_textures: tuple[str, ...] = field(default_factory=tuple)
    eye_colors: tuple[str, ...] = field(default_factory=tuple)
    skin_tones: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class HairTextureConfig(AttributeConfig):
    """Configuration for hair texture attributes."""
    ethnicity_fit: tuple[str, ...] = field(default_factory=tuple)
    prompt_weight: float = 1.0


@dataclass(frozen=True)
class AgeRangeConfig(AttributeConfig):
    """Configuration for age range attributes."""
    age_min: int = 18
    age_max: int = 80
    age_features: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SettingConfig(AttributeConfig):
    """Configuration for setting attributes."""
    lighting_suggestions: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PoseConfig(AttributeConfig):
    """Configuration for pose attributes."""
    complexity: Literal["low", "medium", "high"] = "low"
    explicit: bool = False


@dataclass(frozen=True)
class ClothingConfig(AttributeConfig):
    """Configuration for clothing attributes."""
    pass


@dataclass(frozen=True)
class ActivityConfig(AttributeConfig):
    """Configuration for activity attributes."""
    complexity: Literal["low", "medium", "high"] = "low"
    explicit: bool = False


@dataclass(frozen=True)
class ArtisticStyleConfig(AttributeConfig):
    """Configuration for artistic style attributes."""
    pass


@dataclass(frozen=True)
class FantasyRaceConfig(AttributeConfig):
    """Configuration for fantasy race attributes."""
    pass


@dataclass(frozen=True)
class SpecialEffectConfig(AttributeConfig):
    """Configuration for special effect attributes."""
    pass


@dataclass(frozen=True)
class CosplayStyleConfig(AttributeConfig):
    """Configuration for cosplay style attributes."""
    pass


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


@dataclass(frozen=True)
class RandomizationRules:
    """Rules for randomization and diversity."""
    min_non_white_percentage: float = 0.60
    min_dark_skin_percentage: float = 0.25
    min_age_40_percentage: float = 0.40
    max_young_percentage: float = 0.20


# ============================================================================
# CONCEPT CATEGORIES CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class ConceptCategoriesConfig:
    """Concept categories for prompt analysis."""
    character: tuple[str, ...]
    style: tuple[str, ...]
    content: tuple[str, ...]
    setting: tuple[str, ...]
    quality: tuple[str, ...]
    lighting: tuple[str, ...]
    camera: tuple[str, ...]
    technical: tuple[str, ...]
    subjects: tuple[str, ...]
    age_attributes: tuple[str, ...]
    relationships: tuple[str, ...]
    youth_indicators: tuple[str, ...]
    adult_indicators: tuple[str, ...]
    medical_conditions: tuple[str, ...]
    anatomy: tuple[str, ...]
    physical_details: tuple[str, ...]
    activity: tuple[str, ...]
    clothing: tuple[str, ...]


# ============================================================================
# LORA FILTERS CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class ScoringWeights:
    """Scoring weights for LoRA recommendations."""
    priority_score_weight: float = 0.25
    anatomy_score_weight: float = 0.20
    keyword_score_weight: float = 0.25
    tag_score_weight: float = 0.20
    popularity_score_weight: float = 0.10


@dataclass(frozen=True)
class LoraLimits:
    """Limits for LoRA usage."""
    max_loras: int = 3
    min_confidence: float = 0.5
    max_total_weight: float = 3.0
    max_individual_weight: float = 1.2
    min_individual_weight: float = 0.3


@dataclass(frozen=True)
class LoraFiltersConfig:
    """LoRA filtering configuration."""
    blocked_tags: tuple[str, ...]
    priority_tags: tuple[str, ...]
    anatomy_tags: tuple[str, ...]
    scoring_weights: ScoringWeights
    lora_limits: LoraLimits


# ============================================================================
# PROMPTING STRATEGIES CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class PromptStructure:
    """Structure and ordering for prompts."""
    order: tuple[str, ...]
    required_sections: tuple[str, ...]
    optional_sections: tuple[str, ...]
    ethnicity_weight: float = 1.0
    skin_tone_weight: float = 1.2
    age_weight: float = 1.2
    explicit_pose_weight: float = 1.3
    hair_texture_weight: float = 1.1


@dataclass(frozen=True)
class ModelStrategy:
    """Strategy configuration for specific model types."""
    default_sampler: str
    default_clip_skip: int
    resolution_preference: Literal["landscape", "portrait", "square"]
    steps_multiplier: float = 1.0
    cfg_multiplier: float = 1.0


@dataclass(frozen=True)
class NegativePrompts:
    """Negative prompts for different scenarios."""
    photorealistic: tuple[str, ...]
    explicit: tuple[str, ...]
    age_inappropriate: tuple[str, ...]


@dataclass(frozen=True)
class PromptingStrategiesConfig:
    """Prompting strategies configuration."""
    prompt_structure: PromptStructure
    model_strategies: dict[str, ModelStrategy]
    negative_prompts: NegativePrompts


# ============================================================================
# GENERATION PROFILES CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class AgeProfile:
    """Age profile configuration."""
    min_age: int
    max_age: int
    default_weight: float
    probability: float
    features: tuple[str, ...]


@dataclass(frozen=True)
class GroupProfile:
    """Group profile configuration."""
    subjects: int
    probability: float
    default_resolution: tuple[int, int]
    description: str


@dataclass(frozen=True)
class ActivityProfile:
    """Activity profile configuration."""
    complexity: Literal["low", "medium", "high"]
    default_steps: int
    default_cfg: float
    probability: float


@dataclass(frozen=True)
class DetailPreset:
    """Detail preset configuration."""
    base_steps: int
    base_cfg: float
    base_resolution: tuple[int, int]
    description: str


@dataclass(frozen=True)
class DefaultRanges:
    """Default ranges for generation parameters."""
    min_steps: int = 20
    max_steps: int = 80
    min_cfg: float = 7.0
    max_cfg: float = 15.0
    min_resolution: tuple[int, int] = (768, 768)
    max_resolution: tuple[int, int] = (1536, 1536)
    min_clip_skip: int = 1
    max_clip_skip: int = 12


@dataclass(frozen=True)
class VramPreset:
    """VRAM preset configuration."""
    max_resolution: tuple[int, int]
    max_steps: int
    max_cfg: float
    estimated_vram: float
    description: str


@dataclass(frozen=True)
class GenerationProfilesConfig:
    """Generation profiles configuration."""
    age_profiles: dict[str, AgeProfile]
    group_profiles: dict[str, GroupProfile]
    activity_profiles: dict[str, ActivityProfile]
    detail_presets: dict[str, DetailPreset]
    default_ranges: DefaultRanges
    vram_presets: dict[str, VramPreset]


# ============================================================================
# MAIN CONFIGURATION CLASS
# ============================================================================

@dataclass(frozen=True)
class PromptingConfig:
    """Complete prompting configuration combining all sub-configurations."""
    character_attributes: CharacterAttributesConfig
    randomization_rules: RandomizationRules
    concept_categories: ConceptCategoriesConfig
    lora_filters: LoraFiltersConfig
    prompting_strategies: PromptingStrategiesConfig
    generation_profiles: GenerationProfilesConfig
