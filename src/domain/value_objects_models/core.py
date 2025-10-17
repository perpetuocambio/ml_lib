"""Core types for character attribute definitions and prompting.

This module consolidates core type definitions that were previously in
intelligent/prompting/core/. These types are fundamental to the character
generation and prompting systems.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List


@dataclass
class AttributeRelation:
    """Represents a relationship between attributes (compatibility, requirement, etc.)."""

    attribute_type: "AttributeType"
    attribute_name: str

    def matches(self, attr_type: "AttributeType", attr_name: str) -> bool:
        """Check if this relation matches the given attribute."""
        return (self.attribute_type == attr_type and
                (self.attribute_name == attr_name or self.attribute_name is None))


class AttributeType(Enum):
    """Types of character attributes."""

    # Core identity
    AGE_RANGE = "age_ranges"
    ETHNICITY = "ethnicities"
    SKIN_TONE = "skin_tones"

    # Physical features
    EYE_COLOR = "eye_colors"
    HAIR_COLOR = "hair_colors"
    HAIR_TEXTURE = "hair_textures"
    BODY_TYPE = "body_types"
    BREAST_SIZE = "breast_sizes"

    # Appearance details
    PHYSICAL_FEATURE = "physical_features"
    BODY_SIZE = "body_sizes"
    AESTHETIC_STYLE = "aesthetic_styles"
    FANTASY_RACE = "fantasy_races"

    # Clothing and accessories
    CLOTHING_STYLE = "clothing_styles"
    CLOTHING_CONDITION = "clothing_conditions"
    CLOTHING_DETAIL = "clothing_details"
    ACCESSORY = "accessories"

    # Activities and context
    ACTIVITY = "activities"
    EMOTIONAL_STATE = "emotional_states"
    POSE = "poses"
    ENVIRONMENT = "environments"
    SETTING = "settings"

    # Special attributes
    COSPLAY_STYLE = "cosplay_styles"
    ERATIC_TOY = "erotic_toys"
    WEATHER_CONDITION = "weather_conditions"
    SPECIAL_EFFECT = "special_effects"
    ARTISTIC_STYLE = "artistic_styles"


@dataclass
class AttributeDefinition:
    """Definition of a character attribute with all its properties."""

    # Basic properties
    name: str
    attribute_type: AttributeType
    keywords: List[str] = field(default_factory=list)
    probability: float = 1.0
    prompt_weight: float = 1.0

    # Age restrictions
    min_age: int = 18
    max_age: int = 80

    # Compatibility and relationships
    compatible_with: List[AttributeRelation] = field(default_factory=list)
    incompatible_with: List[AttributeRelation] = field(default_factory=list)
    requires: List[AttributeRelation] = field(default_factory=list)

    # Special properties for different attribute types
    ethnicity_associations: List[str] = field(default_factory=list)
    lighting_suggestions: List[str] = field(default_factory=list)
    complexity: str = "medium"
    explicit: bool = False

    # Metadata and flags
    is_blocked: bool = False  # For content safety
    is_recommended: bool = False  # For preferred combinations
    metadata: dict[str, str] = field(default_factory=dict)  # String metadata only

    def __post_init__(self):
        """Initialize default values."""
        # Ensure lists are properly initialized
        if self.keywords is None:
            self.keywords = []
        if self.ethnicity_associations is None:
            self.ethnicity_associations = []
        if self.lighting_suggestions is None:
            self.lighting_suggestions = []
        if self.compatible_with is None:
            self.compatible_with = []
        if self.incompatible_with is None:
            self.incompatible_with = []
        if self.requires is None:
            self.requires = []
        if self.metadata is None:
            self.metadata = {}

    def is_compatible_with(self, other: 'AttributeDefinition') -> bool:
        """Check if this attribute is compatible with another."""
        # Check explicit incompatibilities
        for relation in self.incompatible_with:
            if relation.matches(other.attribute_type, other.name):
                return False

        for relation in other.incompatible_with:
            if relation.matches(self.attribute_type, self.name):
                return False

        return True

    def validate_age(self, age: int) -> bool:
        """Validate if this attribute is appropriate for given age."""
        return self.min_age <= age <= self.max_age

    def get_prompt_segment(self) -> str:
        """Generate prompt segment for this attribute."""
        if not self.keywords:
            return ""

        keywords_str = ", ".join(self.keywords)
        if self.prompt_weight != 1.0:
            return f"({keywords_str}:{self.prompt_weight})"
        else:
            return keywords_str
