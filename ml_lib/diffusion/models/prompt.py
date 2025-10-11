"""Prompt analysis entities."""

from dataclasses import dataclass, field
from enum import Enum


class ArtisticStyle(Enum):
    """Detected artistic style."""
    PHOTOREALISTIC = "photorealistic"
    ANIME = "anime"
    CARTOON = "cartoon"
    PAINTING = "painting"
    SKETCH = "sketch"
    ABSTRACT = "abstract"
    CONCEPT_ART = "concept_art"
    UNKNOWN = "unknown"


class ContentType(Enum):
    """Type of content being generated."""
    CHARACTER = "character"
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    SCENE = "scene"
    OBJECT = "object"
    ABSTRACT = "abstract"
    UNKNOWN = "unknown"


class QualityLevel(Enum):
    """Desired quality level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MASTERPIECE = "masterpiece"


@dataclass
class Intent:
    """Detected intent from prompt analysis."""

    artistic_style: ArtisticStyle
    content_type: ContentType
    quality_level: QualityLevel
    confidence: float = 0.0

    def __post_init__(self):
        """Validate intent."""
        if isinstance(self.artistic_style, str):
            self.artistic_style = ArtisticStyle(self.artistic_style)
        if isinstance(self.content_type, str):
            self.content_type = ContentType(self.content_type)
        if isinstance(self.quality_level, str):
            self.quality_level = QualityLevel(self.quality_level)

        assert 0.0 <= self.confidence <= 1.0, "Confidence must be between 0 and 1"


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


class ComplexityCategory(Enum):
    """Complexity category for prompts."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class PromptAnalysis:
    """Result of prompt analysis."""

    original_prompt: str
    tokens: list[str] = field(default_factory=list)
    detected_concepts: dict[str, list[str]] = field(default_factory=dict)
    intent: Intent | None = None
    complexity_score: float = 0.0
    emphasis_map: dict[str, float] = field(default_factory=dict)

    @property
    def complexity_category(self) -> ComplexityCategory:
        """Get complexity category from score."""
        if self.complexity_score < 0.3:
            return ComplexityCategory.SIMPLE
        elif self.complexity_score < 0.7:
            return ComplexityCategory.MODERATE
        else:
            return ComplexityCategory.COMPLEX

    @property
    def concept_count(self) -> int:
        """Total number of detected concepts."""
        return sum(len(concepts) for concepts in self.detected_concepts.values())

    def get_concepts_by_category(self, category: str) -> list[str]:
        """Get concepts for a specific category."""
        return self.detected_concepts.get(category, [])

    def has_concept(self, concept: str) -> bool:
        """Check if a concept is present."""
        for concepts in self.detected_concepts.values():
            if concept.lower() in [c.lower() for c in concepts]:
                return True
        return False


# ============================================================================
# Character Attribute Entities (from intelligent/prompting/entities/character_attribute.py)
# ============================================================================


@dataclass
class AttributeConfig:
    """Configuration for a character attribute."""

    keywords: list[str]
    probability: float = 1.0
    prompt_weight: float = 1.0
    ethnicity_associations: list[str] | None = None
    min_age: int = 18
    max_age: int = 80
    ethnicity_fit: list[str] | None = None
    age_features: list[str] | None = None
    lighting_suggestions: list[str] | None = None
    complexity: str = "medium"
    explicit: bool = False
    age_min: int | None = None
    age_max: int | None = None

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

    skin_tones: dict[str, AttributeConfig]
    ethnicities: dict[str, AttributeConfig]
    eye_colors: dict[str, AttributeConfig]
    hair_colors: dict[str, AttributeConfig]
    hair_textures: dict[str, AttributeConfig]
    body_types: dict[str, AttributeConfig]
    breast_sizes: dict[str, AttributeConfig]
    age_ranges: dict[str, AttributeConfig]
    settings: dict[str, AttributeConfig]
    poses: dict[str, AttributeConfig]
    clothing_styles: dict[str, AttributeConfig]
    clothing_conditions: dict[str, AttributeConfig]
    clothing_details: dict[str, AttributeConfig]
    cosplay_styles: dict[str, AttributeConfig]
    accessories: dict[str, AttributeConfig]
    erotic_toys: dict[str, AttributeConfig]
    activities: dict[str, AttributeConfig]
    weather_conditions: dict[str, AttributeConfig]
    emotional_states: dict[str, AttributeConfig]
    environment_details: dict[str, AttributeConfig]
    artistic_styles: dict[str, AttributeConfig]
    physical_features: dict[str, AttributeConfig]
    body_sizes: dict[str, AttributeConfig]
    aesthetic_styles: dict[str, AttributeConfig]
    fantasy_races: dict[str, AttributeConfig]
    special_effects: dict[str, AttributeConfig]
    randomization_rules: dict  # Specific typed rules


@dataclass
class GeneratedCharacter:
    """A generated character with all attributes."""

    # Core identity
    age: int
    age_keywords: list[str]

    # Ethnicity and skin (MUST be consistent)
    skin_tone: str
    skin_keywords: list[str]
    skin_prompt_weight: float
    ethnicity: str
    ethnicity_keywords: list[str]
    ethnicity_prompt_weight: float

    # Artistic style
    artistic_style: str
    artistic_keywords: list[str]

    # Physical features
    eye_color: str
    eye_keywords: list[str]
    hair_color: str
    hair_keywords: list[str]
    hair_texture: str
    hair_texture_keywords: list[str]
    hair_texture_weight: float

    # Body
    body_type: str
    body_keywords: list[str]
    breast_size: str
    breast_keywords: list[str]

    # Body size/shape
    body_size: str
    body_size_keywords: list[str]

    # Physical features (freckles, tattoos, etc.)
    physical_features: str
    physical_feature_keywords: list[str]

    # Clothing
    clothing_style: str
    clothing_keywords: list[str]

    # Clothing condition
    clothing_condition: str
    clothing_condition_keywords: list[str]

    # Clothing details
    clothing_details: str
    clothing_detail_keywords: list[str]

    # Aesthetic style (goth, punk, etc.)
    aesthetic_style: str
    aesthetic_keywords: list[str]

    # Fantasy race/characteristics
    fantasy_race: str
    fantasy_race_keywords: list[str]

    # Special effects (wet, cum, sticky, etc.)
    special_effects: str
    special_effect_keywords: list[str]

    # Cosplay style (if applicable)
    cosplay_style: str
    cosplay_keywords: list[str]

    # Accessories
    accessories: list[str]
    accessory_keywords: list[str]

    # Erotic toys (if applicable)
    erotic_toys: list[str]
    toy_keywords: list[str]

    # Activities
    activity: str
    activity_keywords: list[str]

    # Weather conditions
    weather: str
    weather_keywords: list[str]

    # Emotional state/expressions
    emotional_state: str
    emotional_keywords: list[str]

    # Environment details
    environment: str
    environment_keywords: list[str]

    # Scene
    setting: str
    setting_keywords: list[str]
    lighting_suggestions: list[str]
    pose: str
    pose_keywords: list[str]
    pose_complexity: str
    pose_explicit: bool

    # Age-related features
    age_features: list[str]

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
        from dataclasses import asdict
        result = asdict(self)
        result["prompt"] = self.to_prompt()
        return result


# ============================================================================
# LoRA Recommendation Entities (from intelligent/prompting/entities/lora_recommendation.py)
# ============================================================================


@dataclass
class LoRARecommendation:
    """Recommendation for a LoRA."""

    lora_name: str
    lora_metadata: "ModelMetadata"  # Forward reference to avoid circular import
    confidence_score: float
    suggested_alpha: float
    matching_concepts: list[str] = field(default_factory=list)
    reasoning: str = ""

    def __post_init__(self):
        """Validate recommendation."""
        assert 0.0 <= self.confidence_score <= 1.0, "Confidence must be between 0 and 1"
        assert 0.0 < self.suggested_alpha <= 2.0, "Alpha should be between 0 and 2"

    def is_compatible_with(self, other: "LoRARecommendation") -> bool:
        """
        Check compatibility with another LoRA.

        Args:
            other: Another LoRA recommendation

        Returns:
            True if compatible
        """
        # Check for style conflicts
        style_keywords = ["anime", "photorealistic", "cartoon", "3d", "realistic"]

        self_styles = [kw for kw in style_keywords if kw in self.lora_name.lower()]
        other_styles = [kw for kw in style_keywords if kw in other.lora_name.lower()]

        # If both have conflicting styles, not compatible
        if self_styles and other_styles:
            if set(self_styles).isdisjoint(set(other_styles)):
                return False

        # Check base model compatibility
        if self.lora_metadata.base_model != other.lora_metadata.base_model:
            return False

        return True
