"""Consolidated enums for diffusion models."""

from .base import BasePromptEnum
from .physical import (
    SkinTone,
    EyeColor,
    HairTexture,
    HairColor,
    Ethnicity,
    PhysicalFeature,
    BodyType,
    BreastSize,
    AgeRange,
    BodySize,
)
from .style_and_meta import (
    SafetyLevel,
    CharacterFocus,
    QualityTarget,
)

__all__ = [
    "BasePromptEnum",
    "SkinTone",
    "EyeColor",
    "HairTexture",
    "HairColor",
    "Ethnicity",
    "PhysicalFeature",
    "BodyType",
    "BreastSize",
    "AgeRange",
    "BodySize",
    "SafetyLevel",
    "CharacterFocus",
    "QualityTarget",
]
