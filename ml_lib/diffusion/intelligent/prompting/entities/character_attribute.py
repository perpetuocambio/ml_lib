"""Character attribute entities for configurable character generation."""

from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional


@dataclass
class AttributeConfig:
    """Configuration for a character attribute."""
    
    keywords: List[str]
    probability: float = 1.0
    prompt_weight: float = 1.0
    ethnicity_associations: Optional[List[str]] = None
    min_age: int = 18
    max_age: int = 80
    ethnicity_fit: Optional[List[str]] = None
    age_features: Optional[List[str]] = None
    lighting_suggestions: Optional[List[str]] = None
    complexity: str = "medium"
    explicit: bool = False
    age_min: Optional[int] = None
    age_max: Optional[int] = None
    
    def __post_init__(self):
        if self.ethnicity_associations is None:
            self.ethnicity_associations = []
        if self.ethnicity_fit is None:
            self.ethnicity_fit = []
        if self.age_features is None:
            self.age_features = []
        if self.lighting_suggestions is None:
            self.lighting_suggestions = []


@dataclass
class CharacterAttributeSet:
    """A set of all character attributes that can be configured."""
    
    skin_tones: Dict[str, AttributeConfig]
    ethnicities: Dict[str, AttributeConfig] 
    eye_colors: Dict[str, AttributeConfig]
    hair_colors: Dict[str, AttributeConfig]
    hair_textures: Dict[str, AttributeConfig]
    body_types: Dict[str, AttributeConfig]
    breast_sizes: Dict[str, AttributeConfig]
    age_ranges: Dict[str, AttributeConfig]
    settings: Dict[str, AttributeConfig]
    poses: Dict[str, AttributeConfig]
    clothing_styles: Dict[str, AttributeConfig]
    clothing_conditions: Dict[str, AttributeConfig]
    clothing_details: Dict[str, AttributeConfig]
    cosplay_styles: Dict[str, AttributeConfig]
    accessories: Dict[str, AttributeConfig]
    erotic_toys: Dict[str, AttributeConfig]
    activities: Dict[str, AttributeConfig]
    weather_conditions: Dict[str, AttributeConfig]
    emotional_states: Dict[str, AttributeConfig]
    environment_details: Dict[str, AttributeConfig]
    artistic_styles: Dict[str, AttributeConfig]
    physical_features: Dict[str, AttributeConfig]
    body_sizes: Dict[str, AttributeConfig]
    aesthetic_styles: Dict[str, AttributeConfig]
    fantasy_races: Dict[str, AttributeConfig]
    special_effects: Dict[str, AttributeConfig]
    randomization_rules: dict  # Specific typed rules, not Any


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
        """
        Generate prompt from character attributes.

        Args:
            include_explicit: Include explicit pose keywords

        Returns:
            Formatted prompt string
        """
        parts = []

        # Artistic style (highest priority for visual appearance)
        if self.artistic_keywords and self.artistic_style != "photorealistic":  # Usually photorealistic is default
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
        if self.body_size_keywords and self.body_size != "average":  # Only add if not average
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
            # For original characters, add as descriptive elements
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
            # Add clothing if not already covered by cosplay
            if not self.cosplay_keywords or self.cosplay_style == "original_character":
                clothing_str = ", ".join(self.clothing_keywords)
                parts.append(f"({clothing_str}:1.1)")
        else:
            # If nude, add descriptive nude keywords
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
        if emotional_str.strip():  # Only add if there are actual keywords
            parts.append(f"({emotional_str}:1.05)")

        # Weather conditions
        weather_str = ", ".join(self.weather_keywords)
        if weather_str.strip():  # Only add if there are actual keywords
            parts.append(f"({weather_str}:1.05)")

        # Environment details
        env_str = ", ".join(self.environment_keywords)
        if env_str.strip():  # Only add if there are actual keywords
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
        lighting_str = ", ".join(self.lighting_suggestions[:2])  # Max 2
        parts.append(f"({lighting_str}:1.1)")

        return ", ".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all character attributes plus generated prompt.
        """
        result = asdict(self)
        result["prompt"] = self.to_prompt()
        return result