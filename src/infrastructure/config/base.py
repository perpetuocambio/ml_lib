"""Base configuration dataclasses for the diffusion module."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .types import OptimizationLevel, SafetyLevel, TagList, WeightDict


@dataclass(frozen=True)
class NegativePromptsConfig:
    """Configuration for negative prompts."""
    general: list[str] = field(default_factory=lambda: [
        "low quality", "blurry", "deformed", "bad anatomy"
    ])
    photorealistic: list[str] = field(default_factory=lambda: [
        "cartoon", "anime", "unrealistic"
    ])
    age_inappropriate: list[str] = field(default_factory=lambda: [
        "child", "minor", "teen", "underage"
    ])
    explicit: list[str] = field(default_factory=lambda: ["nsfw"])


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for LoRA recommendations."""
    blocked_tags: TagList = field(default_factory=lambda: [
        "anime", "cartoon", "child", "minor", "teen"
    ])
    priority_tags: TagList = field(default_factory=lambda: [
        "photorealistic", "nsfw", "mature", "realistic"
    ])
    anatomy_tags: TagList = field(default_factory=lambda: [
        "anatomy", "detailed", "breasts", "body"
    ])
    scoring_weights: WeightDict = field(default_factory=lambda: {
        "priority_score_weight": 0.25,
        "anatomy_score_weight": 0.20,
        "keyword_score_weight": 0.25,
        "tag_score_weight": 0.20,
        "popularity_score_weight": 0.10,
    })
    limits: dict[str, int | float] = field(default_factory=lambda: {
        "max_loras": 3,
        "min_confidence": 0.5,
        "min_individual_weight": 0.3,
        "max_individual_weight": 1.2,
        "max_total_weight": 3.0,
    })


@dataclass(frozen=True)
class ParameterRanges:
    """Parameter ranges for optimization."""
    min_steps: int = 20
    max_steps: int = 80
    min_cfg: float = 7.0
    max_cfg: float = 15.0
    min_resolution: tuple[int, int] = (768, 768)
    max_resolution: tuple[int, int] = (1536, 1536)


@dataclass(frozen=True)
class VRAMPreset:
    """VRAM preset configuration."""
    max_resolution: tuple[int, int]
    enable_quantization: bool = True
    enable_vae_tiling: bool = True


@dataclass(frozen=True)
class VRAMConfig:
    """VRAM configuration for different memory levels."""
    low: VRAMPreset = field(default_factory=lambda: VRAMPreset(
        max_resolution=(768, 768)
    ))
    medium: VRAMPreset = field(default_factory=lambda: VRAMPreset(
        max_resolution=(1024, 1024)
    ))
    high: VRAMPreset = field(default_factory=lambda: VRAMPreset(
        max_resolution=(1216, 832)
    ))


@dataclass(frozen=True)
class DetailPreset:
    """Detail preset for generation."""
    base_steps: int
    base_cfg: float
    base_resolution: tuple[int, int]


@dataclass(frozen=True)
class DetailConfig:
    """Detail configuration for different quality levels."""
    low: DetailPreset = field(default_factory=lambda: DetailPreset(
        base_steps=25, base_cfg=7.5, base_resolution=(768, 768)
    ))
    medium: DetailPreset = field(default_factory=lambda: DetailPreset(
        base_steps=35, base_cfg=9.0, base_resolution=(1024, 1024)
    ))
    high: DetailPreset = field(default_factory=lambda: DetailPreset(
        base_steps=50, base_cfg=11.0, base_resolution=(1152, 896)
    ))


@dataclass(frozen=True)
class SamplerConfig:
    """Sampler configuration for different models."""
    default_sampler: str
    default_clip_skip: int = 1
    alternative_samplers: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelStrategies:
    """Model-specific strategies."""
    sdxl: SamplerConfig = field(default_factory=lambda: SamplerConfig(
        default_sampler="DPM++ 2M Karras",
        alternative_samplers=["Euler A", "DPM++ SDE"]
    ))
    sd20: SamplerConfig = field(default_factory=lambda: SamplerConfig(
        default_sampler="DPM++ 2M",
        alternative_samplers=["Euler A"]
    ))
    sd15: SamplerConfig = field(default_factory=lambda: SamplerConfig(
        default_sampler="Euler A",
        alternative_samplers=["DPM++ 2M"]
    ))


@dataclass(frozen=True)
class GroupProfile:
    """Profile for group generation (couples, trios)."""
    default_resolution: tuple[int, int]
    min_spacing: int = 64


@dataclass(frozen=True)
class GroupProfiles:
    """Profiles for different group sizes."""
    single: GroupProfile = field(default_factory=lambda: GroupProfile(
        default_resolution=(896, 1152)
    ))
    couple: GroupProfile = field(default_factory=lambda: GroupProfile(
        default_resolution=(1024, 1024)
    ))
    trio: GroupProfile = field(default_factory=lambda: GroupProfile(
        default_resolution=(1280, 896)
    ))


@dataclass(frozen=True)
class ConceptCategories:
    """Concept categories for prompt analysis."""
    character: list[str] = field(default_factory=lambda: [
        "woman", "man", "person", "character"
    ])
    style: list[str] = field(default_factory=lambda: [
        "photorealistic", "anime", "cartoon", "realistic"
    ])
    content: list[str] = field(default_factory=lambda: [
        "portrait", "scene", "landscape"
    ])
    quality: list[str] = field(default_factory=lambda: [
        "masterpiece", "high quality", "detailed"
    ])


@dataclass(frozen=True)
class DiffusionConfig:
    """Main configuration for the diffusion module.

    This is the central configuration object that contains all sub-configurations.
    All fields are frozen to ensure immutability and thread-safety.
    """

    # Sub-configurations
    negative_prompts: NegativePromptsConfig = field(default_factory=NegativePromptsConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    parameter_ranges: ParameterRanges = field(default_factory=ParameterRanges)
    vram: VRAMConfig = field(default_factory=VRAMConfig)
    detail: DetailConfig = field(default_factory=DetailConfig)
    model_strategies: ModelStrategies = field(default_factory=ModelStrategies)
    group_profiles: GroupProfiles = field(default_factory=GroupProfiles)
    concept_categories: ConceptCategories = field(default_factory=ConceptCategories)

    # General settings
    safety_level: SafetyLevel = SafetyLevel.STRICT
    optimization_level: OptimizationLevel = OptimizationLevel.BALANCED
    cache_dir: Optional[Path] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate ranges
        if self.parameter_ranges.min_steps >= self.parameter_ranges.max_steps:
            raise ValueError("min_steps must be less than max_steps")
        if self.parameter_ranges.min_cfg >= self.parameter_ranges.max_cfg:
            raise ValueError("min_cfg must be less than max_cfg")


__all__ = [
    "NegativePromptsConfig",
    "LoRAConfig",
    "ParameterRanges",
    "VRAMPreset",
    "VRAMConfig",
    "DetailPreset",
    "DetailConfig",
    "SamplerConfig",
    "ModelStrategies",
    "GroupProfile",
    "GroupProfiles",
    "ConceptCategories",
    "DiffusionConfig",
]
