"""
Generation parameter value objects.

This module provides type-safe classes for generation parameters,
replacing dict[str, Any] usage throughout the codebase.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GenerationParameters:
    """Type-safe generation parameters replacing dict[str, Any]."""

    num_steps: int
    guidance_scale: float
    width: int
    height: int
    sampler: str
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.num_steps < 1 or self.num_steps > 150:
            raise ValueError(f"num_steps must be 1-150, got {self.num_steps}")
        if self.guidance_scale < 1.0 or self.guidance_scale > 30.0:
            raise ValueError(f"guidance_scale must be 1.0-30.0, got {self.guidance_scale}")
        if self.width < 64 or self.width > 4096:
            raise ValueError(f"width must be 64-4096, got {self.width}")
        if self.height < 64 or self.height > 4096:
            raise ValueError(f"height must be 64-4096, got {self.height}")


@dataclass(frozen=True)
class ParameterModification:
    """Record of a parameter being modified by user."""

    recommended_value: int | float | str
    actual_value: int | float | str


@dataclass(frozen=True)
class ParameterModifications:
    """Collection of parameter modifications replacing dict[str, dict]."""

    num_steps: Optional[ParameterModification] = None
    guidance_scale: Optional[ParameterModification] = None
    width: Optional[ParameterModification] = None
    height: Optional[ParameterModification] = None
    sampler: Optional[ParameterModification] = None
    seed: Optional[ParameterModification] = None

    def has_modifications(self) -> bool:
        """Check if any parameters were modified."""
        return any([
            self.num_steps is not None,
            self.guidance_scale is not None,
            self.width is not None,
            self.height is not None,
            self.sampler is not None,
            self.seed is not None,
        ])

    def count(self) -> int:
        """Count number of modifications."""
        return sum([
            1 for mod in [
                self.num_steps,
                self.guidance_scale,
                self.width,
                self.height,
                self.sampler,
                self.seed,
            ]
            if mod is not None
        ])


@dataclass(frozen=True)
class FeedbackStatistics:
    """Feedback statistics replacing dict[str, Any]."""

    total_feedback: int
    average_rating: float
    like_rate: float
    saved_count: int
    shared_count: int
    rating_1_count: int
    rating_2_count: int
    rating_3_count: int
    rating_4_count: int
    rating_5_count: int


@dataclass(frozen=True)
class TagCount:
    """A tag with its occurrence count."""

    tag: str
    count: int
