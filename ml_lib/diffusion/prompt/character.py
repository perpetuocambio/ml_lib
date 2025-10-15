"""Character models - consolidated from prompting package."""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

from ml_lib.diffusion.prompt.core import AttributeDefinition, AttributeType
from ml_lib.diffusion.models.enums import (
    SafetyLevel,
    CharacterFocus,
    QualityTarget,
)


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
            parts.append(
                f"{hair_color_str}, ({hair_texture_str}:{self.hair_texture_weight})"
            )
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

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary with all character attributes plus generated prompt.
        """
        result = asdict(self)
        result["prompt"] = self.to_prompt()
        return result


@dataclass
class ValidationResult:
    """Result of character attribute validation.

    This class replaces the untyped Dict[str, Any] that was previously returned
    by validate_character_selection().

    All fields are strongly typed and documented.
    """

    is_valid: bool
    """Overall validation status - True if all checks passed."""

    compatibility_valid: bool
    """Whether selected attributes are compatible with each other."""

    issues: list[str] = field(default_factory=list)
    """Complete list of all validation issues found."""

    age_consistency_issues: list[str] = field(default_factory=list)
    """Issues related to age consistency (e.g., age-inappropriate attributes)."""

    blocked_content_issues: list[str] = field(default_factory=list)
    """Issues related to blocked/inappropriate content."""

    suggestions: list[str] = field(default_factory=list)
    """Suggestions for resolving validation issues."""

    @property
    def total_issue_count(self) -> int:
        """Total number of issues found."""
        return len(self.issues)

    @property
    def has_blocking_issues(self) -> bool:
        """Whether there are any blocking issues that prevent generation."""
        return len(self.blocked_content_issues) > 0

    @property
    def has_warnings(self) -> bool:
        """Whether there are non-blocking warnings."""
        return not self.is_valid and not self.has_blocking_issues


@dataclass
class CompatibilityMap:
    """Map of compatible attributes by type.

    This class replaces Dict[AttributeType, List[AttributeDefinition]] that was
    previously used to represent compatibility information.

    Uses a list of tuples internally for type safety, with helper methods for access.
    """

    _entries: list[tuple[AttributeType, AttributeDefinition]] = field(
        default_factory=list
    )

    def add(self, attr_type: AttributeType, attribute: AttributeDefinition) -> None:
        """Add a compatible attribute for a specific type."""
        self._entries.append((attr_type, attribute))

    def get_by_type(self, attr_type: AttributeType) -> list[AttributeDefinition]:
        """Get all compatible attributes for a specific type."""
        return [attr for t, attr in self._entries if t == attr_type]

    def has_type(self, attr_type: AttributeType) -> bool:
        """Check if there are any compatible attributes for a specific type."""
        return any(t == attr_type for t, _ in self._entries)

    def all_types(self) -> set[AttributeType]:
        """Get all attribute types that have compatible attributes."""
        return {t for t, _ in self._entries}

    def all_attributes(self) -> list[AttributeDefinition]:
        """Get all compatible attributes across all types."""
        return [attr for _, attr in self._entries]

    @property
    def is_empty(self) -> bool:
        """Whether this compatibility map has any entries."""
        return len(self._entries) == 0

    @property
    def count(self) -> int:
        """Total number of compatible attribute entries."""
        return len(self._entries)


@dataclass
class GenerationPreferences:
    """Preferences for character generation.

    This class consolidates the previously separate GenerationPreferences
    and CharacterGenerationContext classes.

    Uses enums instead of string literals for type safety.
    """

    # Targeting
    target_age: int | None = None
    """Specific target age for character (None = random)."""

    target_ethnicity: str | None = None
    """Specific target ethnicity (None = random)."""

    target_style: str | None = None
    """Specific target style (None = random)."""

    # Content control
    explicit_content_allowed: bool = True
    """Whether explicit/NSFW content is allowed."""

    safety_level: SafetyLevel = SafetyLevel.STRICT
    """Content filtering safety level."""

    # Visual preferences
    character_focus: CharacterFocus = CharacterFocus.PORTRAIT
    """Framing/composition focus for the character."""

    quality_target: QualityTarget = QualityTarget.HIGH
    """Target quality level for generation."""

    # Diversity
    diversity_target: float = 0.6
    """Target percentage of diverse/non-white characters (0.0-1.0)."""

    def __post_init__(self) -> None:
        """Validate preferences after initialization."""
        if self.diversity_target < 0.0 or self.diversity_target > 1.0:
            raise ValueError(
                f"diversity_target must be between 0 and 1, got {self.diversity_target}"
            )

        if self.target_age is not None and (
            self.target_age < 18 or self.target_age > 100
        ):
            raise ValueError(
                f"target_age must be between 18 and 100, got {self.target_age}"
            )
