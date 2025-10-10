"""Generated character model."""

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class GeneratedCharacter:
    """A generated character with all attributes."""

    # Core identity
    age: int
    age_keywords: List[str]

    # Ethnicity and skin (MUST be consistent)
    skin_tone: str
    skin_keywords: List[str]
    skin_prompt_weight: float
    ethnicity: str
    ethnicity_keywords: List[str]
    ethnicity_prompt_weight: float

    # Artistic style
    artistic_style: str
    artistic_keywords: List[str]

    # Physical features
    eye_color: str
    eye_keywords: List[str]
    hair_color: str
    hair_keywords: List[str]
    hair_texture: str
    hair_texture_keywords: List[str]
    hair_texture_weight: float

    # Body
    body_type: str
    body_keywords: List[str]
    breast_size: str
    breast_keywords: List[str]

    # Body size/shape
    body_size: str
    body_size_keywords: List[str]

    # Physical features (freckles, tattoos, etc.)
    physical_features: str
    physical_feature_keywords: List[str]

    # Clothing
    clothing_style: str
    clothing_keywords: List[str]

    # Clothing condition
    clothing_condition: str
    clothing_condition_keywords: List[str]

    # Clothing details
    clothing_details: str
    clothing_detail_keywords: List[str]

    # Aesthetic style (goth, punk, etc.)
    aesthetic_style: str
    aesthetic_keywords: List[str]

    # Fantasy race/characteristics
    fantasy_race: str
    fantasy_race_keywords: List[str]

    # Special effects (wet, cum, sticky, etc.)
    special_effects: str
    special_effect_keywords: List[str]

    # Cosplay style (if applicable)
    cosplay_style: str
    cosplay_keywords: List[str]

    # Accessories
    accessories: List[str]
    accessory_keywords: List[str]

    # Erotic toys (if applicable)
    erotic_toys: List[str]
    toy_keywords: List[str]

    # Activities
    activity: str
    activity_keywords: List[str]

    # Weather conditions
    weather: str
    weather_keywords: List[str]

    # Emotional state/expressions
    emotional_state: str
    emotional_keywords: List[str]

    # Environment details
    environment: str
    environment_keywords: List[str]

    # Scene
    setting: str
    setting_keywords: List[str]
    lighting_suggestions: List[str]
    pose: str
    pose_keywords: List[str]
    pose_complexity: str
    pose_explicit: bool

    # Age-related features
    age_features: List[str]

    def to_prompt(self, include_explicit: bool = True) -> str:
        """Generate prompt from character attributes.

        Args:
            include_explicit: Include explicit pose keywords

        Returns:
            Formatted prompt string
        """
        parts = []

        # Artistic style (highest priority for visual appearance)
        if self.artistic_keywords and self.artistic_style != "photorealistic":
            artistic_str = ", ".join(self.artistic_keywords)
            parts.append(f"({artistic_str}:1.1)")

        # Age (weighted)
        age_str = ", ".join(self.age_keywords)
        parts.append(f"({age_str}:1.2)")

        # Ethnicity (HIGH weight to counter bias)
        ethnicity_str = ", ".join(self.ethnicity_keywords)
        parts.append(f"({ethnicity_str}:{self.ethnicity_prompt_weight})")

        # Skin tone (HIGH weight to counter bias)
        skin_str = ", ".join(self.skin_keywords)
        parts.append(f"({skin_str}:{self.skin_prompt_weight})")

        # Physical features (freckles, tattoos, etc.)
        if self.physical_feature_keywords:
            physical_str = ", ".join(self.physical_feature_keywords)
            parts.append(f"({physical_str}:1.05)")

        # Body size/shape
        if self.body_size_keywords and self.body_size != "average":
            body_size_str = ", ".join(self.body_size_keywords)
            parts.append(f"({body_size_str}:1.05)")

        # Eye color
        eye_str = ", ".join(self.eye_keywords)
        parts.append(eye_str)

        # Hair (color + texture with weight for non-straight hair)
        hair_color_str = ", ".join(self.hair_keywords)
        hair_texture_str = ", ".join(self.hair_texture_keywords)

        if self.hair_texture_weight > 1.0:
            parts.append(f"{hair_color_str}, ({hair_texture_str}:{self.hair_texture_weight})")
        else:
            parts.append(f"{hair_color_str}, {hair_texture_str}")

        # Age features
        if self.age_features:
            age_features_str = ", ".join(self.age_features)
            parts.append(f"({age_features_str}:1.1)")

        # Fantasy race/characteristics
        if self.fantasy_race_keywords:
            fantasy_str = ", ".join(self.fantasy_race_keywords)
            parts.append(f"({fantasy_str}:1.1)")

        # Cosplay costume (if applicable)
        if self.cosplay_keywords and self.cosplay_style != "original_character":
            cosplay_str = ", ".join(self.cosplay_keywords)
            parts.append(f"({cosplay_str}:1.15)")
        elif self.cosplay_keywords:
            cosplay_str = ", ".join(self.cosplay_keywords)
            parts.append(f"({cosplay_str}:1.1)")

        # Clothing condition (if applicable)
        if self.clothing_condition_keywords and self.clothing_condition != "intact":
            condition_str = ", ".join(self.clothing_condition_keywords)
            parts.append(f"({condition_str}:1.1)")

        # Clothing details (if applicable)
        if self.clothing_detail_keywords:
            detail_str = ", ".join(self.clothing_detail_keywords)
            parts.append(f"({detail_str}:1.05)")

        # Clothing (if not nude and not cosplay-specific clothing)
        if self.clothing_style != "nude":
            if not self.cosplay_keywords or self.cosplay_style == "original_character":
                clothing_str = ", ".join(self.clothing_keywords)
                parts.append(f"({clothing_str}:1.1)")
        else:
            nude_parts = [", ".join(self.clothing_keywords)]
            parts.extend(nude_parts)

        # Aesthetic style (goth, punk, etc.)
        if self.aesthetic_keywords:
            aesthetic_str = ", ".join(self.aesthetic_keywords)
            parts.append(f"({aesthetic_str}:1.05)")

        # Body
        body_str = ", ".join(self.body_keywords)
        parts.append(body_str)

        breast_str = ", ".join(self.breast_keywords)
        parts.append(f"({breast_str}:1.2)")

        # Special effects (wet, cum, sticky, etc.)
        if self.special_effect_keywords:
            special_str = ", ".join(self.special_effect_keywords)
            parts.append(f"({special_str}:1.1)")

        # Accessories
        if self.accessory_keywords:
            accessory_str = ", ".join(self.accessory_keywords)
            parts.append(f"({accessory_str}:1.05)")

        # Emotional state/expressions
        emotional_str = ", ".join(self.emotional_keywords)
        if emotional_str.strip():
            parts.append(f"({emotional_str}:1.05)")

        # Weather conditions
        weather_str = ", ".join(self.weather_keywords)
        if weather_str.strip():
            parts.append(f"({weather_str}:1.05)")

        # Environment details
        env_str = ", ".join(self.environment_keywords)
        if env_str.strip():
            parts.append(f"({env_str}:1.05)")

        # Erotic toys (only if explicit content allowed)
        if include_explicit and self.toy_keywords:
            toy_str = ", ".join(self.toy_keywords)
            parts.append(f"({toy_str}:1.1)")

        # Activity (only if explicit content allowed)
        if include_explicit and self.activity_keywords:
            activity_str = ", ".join(self.activity_keywords)
            if self.activity in ["sexual", "bdsm"]:
                parts.append(f"({activity_str}:1.2)")
            else:
                parts.append(activity_str)

        # Pose
        if include_explicit or not self.pose_explicit:
            pose_str = ", ".join(self.pose_keywords)
            if self.pose_complexity == "high":
                parts.append(f"({pose_str}:1.3)")
            elif self.pose_complexity == "medium":
                parts.append(f"({pose_str}:1.2)")
            else:
                parts.append(pose_str)

        # Setting
        setting_str = ", ".join(self.setting_keywords)
        parts.append(setting_str)

        # Lighting
        lighting_str = ", ".join(self.lighting_suggestions[:2])
        parts.append(f"({lighting_str}:1.1)")

        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with character attributes and generated prompt
        """
        return {
            "age": self.age,
            "ethnicity": self.ethnicity,
            "skin_tone": self.skin_tone,
            "artistic_style": self.artistic_style,
            "eye_color": self.eye_color,
            "hair_color": self.hair_color,
            "hair_texture": self.hair_texture,
            "body_type": self.body_type,
            "breast_size": self.breast_size,
            "body_size": self.body_size,
            "physical_features": self.physical_features,
            "clothing_style": self.clothing_style,
            "clothing_condition": self.clothing_condition,
            "clothing_details": self.clothing_details,
            "aesthetic_style": self.aesthetic_style,
            "fantasy_race": self.fantasy_race,
            "special_effects": self.special_effects,
            "cosplay_style": self.cosplay_style,
            "accessories": self.accessories,
            "erotic_toys": self.erotic_toys,
            "activity": self.activity,
            "weather": self.weather,
            "emotional_state": self.emotional_state,
            "environment": self.environment,
            "setting": self.setting,
            "pose": self.pose,
            "prompt": self.to_prompt(),
        }
