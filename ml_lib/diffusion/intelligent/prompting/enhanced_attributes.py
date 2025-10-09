"""Enhanced character attributes using class-based approach."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
import yaml
from pathlib import Path


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
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
        """
        Check if this attribute is compatible with another.
        
        Args:
            other: Other attribute to check
            
        Returns:
            True if compatible, False otherwise
        """
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
        """
        Validate if this attribute is appropriate for given age.
        
        Args:
            age: Age to validate
            
        Returns:
            True if valid for age, False otherwise
        """
        return self.min_age <= age <= self.max_age
    
    def get_prompt_segment(self) -> str:
        """
        Generate prompt segment for this attribute.
        
        Returns:
            Formatted prompt segment
        """
        if not self.keywords:
            return ""
        
        keywords_str = ", ".join(self.keywords)
        if self.prompt_weight != 1.0:
            return f"({keywords_str}:{self.prompt_weight})"
        else:
            return keywords_str


class AttributeCollection:
    """Collection of related attributes of the same type."""
    
    def __init__(self, attribute_type: AttributeType):
        """
        Initialize attribute collection.
        
        Args:
            attribute_type: Type of attributes in this collection
        """
        self.attribute_type = attribute_type
        self.attributes: Dict[str, AttributeDefinition] = {}
    
    def add_attribute(self, attribute: AttributeDefinition):
        """
        Add an attribute to the collection.
        
        Args:
            attribute: Attribute to add
        """
        if attribute.attribute_type != self.attribute_type:
            raise ValueError(f"Attribute type mismatch: expected {self.attribute_type}, got {attribute.attribute_type}")
        
        self.attributes[attribute.name] = attribute
    
    def get_attribute(self, name: str) -> Optional[AttributeDefinition]:
        """
        Get an attribute by name.
        
        Args:
            name: Name of attribute to get
            
        Returns:
            Attribute or None if not found
        """
        return self.attributes.get(name)
    
    def get_compatible_attributes(self, other_attributes: List[AttributeDefinition]) -> List[AttributeDefinition]:
        """
        Get attributes that are compatible with a list of other attributes.
        
        Args:
            other_attributes: List of attributes to check compatibility with
            
        Returns:
            List of compatible attributes
        """
        compatible = []
        for attribute in self.attributes.values():
            is_compatible = True
            for other_attr in other_attributes:
                if not attribute.is_compatible_with(other_attr):
                    is_compatible = False
                    break
            
            if is_compatible:
                compatible.append(attribute)
        
        return compatible
    
    def select_by_probability(self) -> Optional[AttributeDefinition]:
        """
        Select an attribute based on probabilities.
        
        Returns:
            Selected attribute or None if no attributes available
        """
        if not self.attributes:
            return None
        
        # Filter out blocked attributes
        available_attributes = [
            attr for attr in self.attributes.values() 
            if not attr.is_blocked
        ]
        
        if not available_attributes:
            return None
        
        # Weighted selection
        weights = [attr.probability for attr in available_attributes]
        total_weight = sum(weights)
        
        if total_weight <= 0:
            # Equal probability for all
            import random
            return random.choice(available_attributes)
        
        # Normalize weights
        probabilities = [w / total_weight for w in weights]
        
        # Random selection
        import random
        return random.choices(available_attributes, weights=probabilities, k=1)[0]


class CharacterAttributeSet:
    """Complete set of all character attributes organized by type."""
    
    def __init__(self):
        """Initialize character attribute set."""
        self.collections: Dict[AttributeType, AttributeCollection] = {}
        
        # Create collections for each attribute type
        for attr_type in AttributeType:
            self.collections[attr_type] = AttributeCollection(attr_type)
    
    def add_attribute(self, attribute: AttributeDefinition):
        """
        Add an attribute to the appropriate collection.
        
        Args:
            attribute: Attribute to add
        """
        collection = self.collections.get(attribute.attribute_type)
        if collection:
            collection.add_attribute(attribute)
        else:
            # Create collection if it doesn't exist
            collection = AttributeCollection(attribute.attribute_type)
            collection.add_attribute(attribute)
            self.collections[attribute.attribute_type] = collection
    
    def get_collection(self, attribute_type: AttributeType) -> Optional[AttributeCollection]:
        """
        Get a collection by attribute type.
        
        Args:
            attribute_type: Type of collection to get
            
        Returns:
            Collection or None if not found
        """
        return self.collections.get(attribute_type)
    
    def get_attribute(self, attribute_type: AttributeType, name: str) -> Optional[AttributeDefinition]:
        """
        Get an attribute by type and name.
        
        Args:
            attribute_type: Type of attribute
            name: Name of attribute
            
        Returns:
            Attribute or None if not found
        """
        collection = self.get_collection(attribute_type)
        if collection:
            return collection.get_attribute(name)
        return None
    
    def load_from_yaml(self, yaml_path: Path):
        """
        Load attributes from YAML configuration.
        
        Args:
            yaml_path: Path to YAML configuration file
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Process each section in the YAML
        for section_name, section_data in config.items():
            if section_name in ['randomization_rules']:
                continue  # Skip non-attribute sections
            
            # Map section name to AttributeType
            try:
                attr_type = AttributeType(section_name)
            except ValueError:
                # Unknown section, skip
                continue
            
            # Process each attribute in the section
            for attr_name, attr_data in section_data.items():
                attribute = self._create_attribute_from_yaml(attr_name, attr_type, attr_data)
                self.add_attribute(attribute)
    
    def _create_attribute_from_yaml(self, name: str, attr_type: AttributeType, 
                                   data: Dict[str, Any]) -> AttributeDefinition:
        """
        Create an attribute definition from YAML data.
        
        Args:
            name: Name of the attribute
            attr_type: Type of the attribute
            data: YAML data for the attribute
            
        Returns:
            Attribute definition
        """
        # Handle blocked content
        is_blocked = False
        blocked_terms = ['schoolgirl', 'school uniform', 'underage', 'minor', 'child', 'teen']
        
        # Check if this attribute should be blocked
        keywords = data.get('keywords', [])
        if any(blocked_term in ' '.join(keywords).lower() for blocked_term in blocked_terms):
            is_blocked = True
        
        # Create attribute
        attribute = AttributeDefinition(
            name=name,
            attribute_type=attr_type,
            keywords=keywords,
            probability=data.get('probability', 1.0),
            prompt_weight=data.get('prompt_weight', 1.0),
            min_age=data.get('min_age', 18),
            max_age=data.get('max_age', 80),
            ethnicity_associations=data.get('ethnicity_associations', []),
            lighting_suggestions=data.get('lighting_suggestions', []),
            complexity=data.get('complexity', 'medium'),
            explicit=data.get('explicit', False),
            is_blocked=is_blocked,
            metadata=data  # Store original data for reference
        )
        
        return attribute
    
    def validate_compatibility(self, attributes: List[AttributeDefinition]) -> Tuple[bool, List[str]]:
        """
        Validate compatibility of a list of attributes.
        
        Args:
            attributes: List of attributes to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check all pairs for compatibility
        for i in range(len(attributes)):
            for j in range(i + 1, len(attributes)):
                attr_a = attributes[i]
                attr_b = attributes[j]
                
                if not attr_a.is_compatible_with(attr_b):
                    issues.append(f"Incompatible attributes: {attr_a.name} ({attr_a.attribute_type.value}) and {attr_b.name} ({attr_b.attribute_type.value})")
        
        # Check age consistency
        ages = [attr for attr in attributes if attr.attribute_type == AttributeType.AGE_RANGE]
        if len(ages) > 1:
            # Check that all age attributes are consistent
            age_values = [(attr.min_age, attr.max_age) for attr in ages]
            # This could be expanded with more sophisticated age validation
            
        is_valid = len(issues) == 0
        return is_valid, issues


# Example usage and testing
if __name__ == "__main__":
    # Create attribute set
    attribute_set = CharacterAttributeSet()
    
    # Load from YAML (would normally point to actual config file)
    # attribute_set.load_from_yaml(Path("character_attributes.yaml"))
    
    # Create some test attributes manually
    test_age = AttributeDefinition(
        name="milf",
        attribute_type=AttributeType.AGE_RANGE,
        keywords=["milf", "mature woman", "older woman", "experienced"],
        probability=0.4,
        min_age=40,
        max_age=54
    )
    
    test_ethnicity = AttributeDefinition(
        name="caucasian",
        attribute_type=AttributeType.ETHNICITY,
        keywords=["caucasian", "european", "white"],
        probability=0.3,
        ethnicity_associations=["fair", "light", "medium"]
    )
    
    test_skin_tone = AttributeDefinition(
        name="medium",
        attribute_type=AttributeType.SKIN_TONE,
        keywords=["medium skin", "olive skin", "tan skin"],
        probability=1.2,
        prompt_weight=1.2
    )
    
    # Add to set
    attribute_set.add_attribute(test_age)
    attribute_set.add_attribute(test_ethnicity)
    attribute_set.add_attribute(test_skin_tone)
    
    # Test compatibility
    print("Created test attributes:")
    print(f"Age: {test_age.get_prompt_segment()}")
    print(f"Ethnicity: {test_ethnicity.get_prompt_segment()}")
    print(f"Skin tone: {test_skin_tone.get_prompt_segment()}")
    
    # Test compatibility checking
    is_compatible, issues = attribute_set.validate_compatibility([test_age, test_ethnicity, test_skin_tone])
    print(f"\nCompatibility check: {'PASS' if is_compatible else 'FAIL'}")
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  - {issue}")