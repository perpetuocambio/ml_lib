"""Optimized parameter entities."""

from dataclasses import dataclass
from enum import Enum


class Priority(Enum):
    """Optimization priority."""
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"


@dataclass
class OptimizedParameters:
    """Optimized generation parameters."""

    num_steps: int
    guidance_scale: float
    width: int
    height: int
    sampler_name: str
    clip_skip: int = 1

    # Estimations
    estimated_time_seconds: float = 0.0
    estimated_vram_gb: float = 0.0
    estimated_quality_score: float = 0.0

    # Strategy
    optimization_strategy: str = "balanced"
    confidence: float = 0.85

    def __post_init__(self):
        """Validate parameters."""
        assert 1 <= self.num_steps <= 150, "Steps must be between 1 and 150"
        assert 1.0 <= self.guidance_scale <= 30.0, "CFG must be between 1 and 30"
        assert self.width > 0 and self.height > 0, "Dimensions must be positive"
        assert 0 <= self.clip_skip <= 12, "Clip skip must be between 0 and 12"

    @property
    def resolution(self) -> tuple[int, int]:
        """Get resolution as tuple."""
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio."""
        return self.width / self.height if self.height > 0 else 1.0
