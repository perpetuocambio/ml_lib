"""Core attribute definition for character generation."""

from dataclasses import dataclass, field
from typing import List, Tuple, Any, Dict
from enum import Enum

from ml_lib.diffusion.intelligent.prompting.core.attribute_type import AttributeType


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
    compatible_with: List[Tuple[AttributeType, str]] = field(default_factory=list)
    incompatible_with: List[Tuple[AttributeType, str]] = field(default_factory=list)
    requires: List[Tuple[AttributeType, str]] = field(default_factory=list)
    
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
        for incompatible_type, incompatible_name in self.incompatible_with:
            if (other.attribute_type == incompatible_type and 
                (incompatible_name is None or other.name == incompatible_name)):
                return False
        
        for incompatible_type, incompatible_name in other.incompatible_with:
            if (self.attribute_type == incompatible_type and 
                (incompatible_name is None or self.name == incompatible_name)):
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
