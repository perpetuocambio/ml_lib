"""Configuration-related types and enums."""

from enum import Enum
from typing import TypeAlias


class OptimizationLevel(Enum):
    """Optimization level for generation."""
    SPEED = "speed"
    BALANCED = "balanced"
    QUALITY = "quality"


class SafetyLevel(Enum):
    """Safety level for content filtering."""
    STRICT = "strict"
    MODERATE = "moderate"
    RELAXED = "relaxed"


# Type aliases for clarity
SamplerName: TypeAlias = str
ModelStrategy: TypeAlias = dict[str, str | int]
TagList: TypeAlias = list[str]
WeightDict: TypeAlias = dict[str, float]


__all__ = [
    "OptimizationLevel",
    "SafetyLevel",
    "SamplerName",
    "ModelStrategy",
    "TagList",
    "WeightDict",
]
