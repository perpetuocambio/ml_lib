"""Batch generation configuration."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class VariationStrategy(Enum):
    """Strategy for generating variations in batch mode."""

    SEED_VARIATION = "seed_variation"
    """Same parameters, different seeds."""

    PARAM_VARIATION = "param_variation"
    """Vary steps, CFG, etc."""

    LORA_VARIATION = "lora_variation"
    """Try different LoRA combinations."""

    MIXED = "mixed"
    """Mix multiple strategies."""


@dataclass
class BatchConfig:
    """Configuration for batch generation."""

    num_images: int = 4
    """Number of images to generate."""

    variation_strategy: VariationStrategy = VariationStrategy.SEED_VARIATION
    """How to generate variations."""

    parallel_generation: bool = False
    """Whether to generate in parallel (requires more VRAM)."""

    base_seed: Optional[int] = None
    """Base seed for seed variation (None = random)."""

    save_individually: bool = True
    """Whether to save each image as it's generated."""

    output_dir: Optional[str] = None
    """Output directory for batch results (None = don't save)."""

    def __post_init__(self):
        """Validate config."""
        if self.num_images <= 0:
            raise ValueError("num_images must be positive")
        if self.base_seed is not None and self.base_seed < 0:
            raise ValueError("base_seed must be non-negative")
