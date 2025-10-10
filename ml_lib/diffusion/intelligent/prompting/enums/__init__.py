"""Enums for intelligent prompting module.

Organized into logical categories:
- physical: Character physical attributes (ethnicity, skin, hair, body, age)
- appearance: Clothing, accessories, cosplay
- scene: Settings, poses, activities, weather
- style: Artistic styles, aesthetics, effects
- emotional: Emotional states and interactive elements
- meta: Configuration levels (safety, quality, complexity)
"""

from ml_lib.diffusion.intelligent.prompting.enums.base_prompt_enum import BasePromptEnum

# Re-export all submodules for convenience
from ml_lib.diffusion.intelligent.prompting.enums.physical import *
from ml_lib.diffusion.intelligent.prompting.enums.appearance import *
from ml_lib.diffusion.intelligent.prompting.enums.scene import *
from ml_lib.diffusion.intelligent.prompting.enums.style import *
from ml_lib.diffusion.intelligent.prompting.enums.emotional import *
from ml_lib.diffusion.intelligent.prompting.enums.meta import *

__all__ = [
    "BasePromptEnum",
    # Physical attributes
    "Ethnicity",
    "SkinTone",
    "EyeColor",
    "HairColor",
    "HairTexture",
    "BodyType",
    "BodySize",
    "BreastSize",
    "AgeRange",
    "PhysicalFeature",
    # Appearance
    "ClothingStyle",
    "ClothingCondition",
    "ClothingDetail",
    "Accessory",
    "CosplayStyle",
    # Scene
    "Setting",
    "Environment",
    "WeatherCondition",
    "Pose",
    "Activity",
    # Style
    "ArtisticStyle",
    "AestheticStyle",
    "SpecialEffect",
    "FantasyRace",
    # Emotional
    "EmotionalState",
    "EroticToy",
    # Meta
    "SafetyLevel",
    "CharacterFocus",
    "QualityTarget",
    "ComplexityLevel",
]
