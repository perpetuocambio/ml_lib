"""Object-oriented architecture for intelligent character attribute management."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum


class AttributeCategory(Enum):
    """Categories of character attributes."""
    
    # Core identity
    AGE = "age_ranges"
    ETHNICITY = "ethnicities"
    SKIN_TONE = "skin_tones"
    
    # Physical features
    EYE_COLOR = "eye_colors"
    HAIR_COLOR = "hair_colors"
    HAIR_TEXTURE = "hair_textures"
    BODY_TYPE = "body_types"
    BREAST_SIZE = "breast_sizes"
    
    # Appearance details
    PHYSICAL_FEATURES = "physical_features"
    BODY_SIZES = "body_sizes"
    AESTHETIC_STYLES = "aesthetic_styles"
    FANTASY_RACES = "fantasy_races"
    
    # Clothing and accessories
    CLOTHING_STYLES = "clothing_styles"
    CLOTHING_CONDITIONS = "clothing_conditions"
    CLOTHING_DETAILS = "clothing_details"
    ACCESSORIES = "accessories"
    
    # Activities and context
    ACTIVITIES = "activities"
    EMOTIONAL_STATES = "emotional_states"
    POSITIONS = "positions"
    ENVIRONMENTS = "environments"
    SETTINGS = "settings"
    
    # Special attributes
    COSPLAY_STYLES = "cosplay_styles"
    ERATIC_TOYS = "erotic_toys"
    WEATHER = "weather_conditions"
    SPECIAL_EFFECTS = "special_effects"
    ARTISTIC_STYLES = "artistic_styles"


@dataclass
class AttributeConfig:
    """Configuration for a character attribute."""
    
    # Core properties
    name: str
    category: AttributeCategory
    keywords: List[str]
    probability: float = 1.0
    
    # Age restrictions
    min_age: int = 18
    max_age: int = 80
    
    # Visual weight in prompt
    prompt_weight: float = 1.0
    
    # Compatibility rules
    compatible_with: List[Tuple[AttributeCategory, str]] = field(default_factory=list)
    incompatible_with: List[Tuple[AttributeCategory, str]] = field(default_factory=list)
    requires: List[Tuple[AttributeCategory, str]] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class SmartAttribute(ABC):
    """Base class for intelligent character attributes."""
    
    def __init__(self, config: AttributeConfig):
        """
        Initialize smart attribute.
        
        Args:
            config: Attribute configuration
        """
        self.config = config
        self._selected_value: Optional[str] = None
        self._associated_attributes: Dict[AttributeCategory, 'SmartAttribute'] = {}
    
    @property
    def name(self) -> str:
        """Get attribute name."""
        return self.config.name
    
    @property
    def category(self) -> AttributeCategory:
        """Get attribute category."""
        return self.config.category
    
    @property
    def keywords(self) -> List[str]:
        """Get attribute keywords."""
        return self.config.keywords.copy()
    
    @property
    def probability(self) -> float:
        """Get selection probability."""
        return self.config.probability
    
    @property
    def min_age(self) -> int:
        """Get minimum age requirement."""
        return self.config.min_age
    
    @property
    def max_age(self) -> int:
        """Get maximum age requirement."""
        return self.config.max_age
    
    @property
    def prompt_weight(self) -> float:
        """Get prompt weight."""
        return self.config.prompt_weight
    
    @property
    def selected_value(self) -> Optional[str]:
        """Get currently selected value."""
        return self._selected_value
    
    def associate_attribute(self, attribute: 'SmartAttribute'):
        """
        Associate another attribute with this one.
        
        Args:
            attribute: Attribute to associate
        """
        self._associated_attributes[attribute.category] = attribute
    
    def get_associated_attribute(self, category: AttributeCategory) -> Optional['SmartAttribute']:
        """
        Get associated attribute by category.
        
        Args:
            category: Category to look up
            
        Returns:
            Associated attribute or None
        """
        return self._associated_attributes.get(category)
    
    def is_compatible_with(self, other: 'SmartAttribute') -> bool:
        """
        Check if this attribute is compatible with another.
        
        Args:
            other: Other attribute to check
            
        Returns:
            True if compatible, False otherwise
        """
        # Check explicit incompatibilities
        for incompatible_cat, incompatible_name in self.config.incompatible_with:
            if (other.category == incompatible_cat and 
                (incompatible_name is None or other.name == incompatible_name)):
                return False
        
        for incompatible_cat, incompatible_name in other.config.incompatible_with:
            if (self.category == incompatible_cat and 
                (incompatible_name is None or self.name == incompatible_name)):
                return False
        
        # Check explicit compatibilities (if defined)
        if self.config.compatible_with:
            is_compatible = False
            for compatible_cat, compatible_name in self.config.compatible_with:
                if (other.category == compatible_cat and 
                    (compatible_name is None or other.name == compatible_name)):
                    is_compatible = True
                    break
            
            if not is_compatible:
                return False
        
        if other.config.compatible_with:
            is_compatible = False
            for compatible_cat, compatible_name in other.config.compatible_with:
                if (self.category == compatible_cat and 
                    (compatible_name is None or self.name == compatible_name)):
                    is_compatible = True
                    break
            
            if not is_compatible:
                return False
        
        # Check requirements
        for required_cat, required_name in self.config.requires:
            if (other.category == required_cat and 
                (required_name is None or other.name == required_name)):
                return True  # Requirement fulfilled
        
        for required_cat, required_name in other.config.requires:
            if (self.category == required_cat and 
                (required_name is None or self.name == required_name)):
                return True  # Requirement fulfilled
        
        return True  # Compatible by default if no explicit rules
    
    def validate_age(self, age: int) -> bool:
        """
        Validate if attribute is appropriate for given age.
        
        Args:
            age: Age to validate
            
        Returns:
            True if valid for age, False otherwise
        """
        return self.min_age <= age <= self.max_age
    
    @abstractmethod
    def generate_prompt_segment(self) -> str:
        """
        Generate prompt segment for this attribute.
        
        Returns:
            Formatted prompt segment
        """
        pass
    
    @abstractmethod
    def select_appropriate_value(self, context: Dict[str, Any]) -> str:
        """
        Select an appropriate value based on context.
        
        Args:
            context: Generation context
            
        Returns:
            Selected value
        """
        pass


class AgeAttribute(SmartAttribute):
    """Intelligent age attribute."""
    
    def generate_prompt_segment(self) -> str:
        """Generate prompt segment for age."""
        if not self._selected_value:
            return ""
        
        # Apply weight to age keywords
        age_keywords = ", ".join(self.config.keywords)
        return f"({age_keywords}:{self.prompt_weight})"
    
    def select_appropriate_value(self, context: Dict[str, Any]) -> str:
        """Select appropriate age value."""
        # Age is typically determined externally, so we just return the name
        self._selected_value = self.name
        return self.name


class EthnicityAttribute(SmartAttribute):
    """Intelligent ethnicity attribute with skin tone consistency."""
    
    def generate_prompt_segment(self) -> str:
        """Generate prompt segment for ethnicity."""
        if not self._selected_value:
            return ""
        
        # Apply weight to ethnicity keywords
        ethnicity_keywords = ", ".join(self.config.keywords)
        return f"({ethnicity_keywords}:{self.prompt_weight})"
    
    def select_appropriate_value(self, context: Dict[str, Any]) -> str:
        """Select appropriate ethnicity value."""
        self._selected_value = self.name
        return self.name
    
    def get_compatible_skin_tones(self) -> List[str]:
        """
        Get skin tones that are compatible with this ethnicity.
        
        Returns:
            List of compatible skin tone names
        """
        # This would reference the ethnicity associations in the config
        return self.config.metadata.get("skin_tones", [])


class SkinToneAttribute(SmartAttribute):
    """Intelligent skin tone attribute."""
    
    def generate_prompt_segment(self) -> str:
        """Generate prompt segment for skin tone."""
        if not self._selected_value:
            return ""
        
        # Apply weight to skin tone keywords
        skin_keywords = ", ".join(self.config.keywords)
        return f"({skin_keywords}:{self.prompt_weight})"
    
    def select_appropriate_value(self, context: Dict[str, Any]) -> str:
        """Select appropriate skin tone value."""
        self._selected_value = self.name
        return self.name


class ClothingStyleAttribute(SmartAttribute):
    """Intelligent clothing style attribute."""
    
    def generate_prompt_segment(self) -> str:
        """Generate prompt segment for clothing style."""
        if not self._selected_value:
            return ""
        
        # Apply weight to clothing keywords
        clothing_keywords = ", ".join(self.config.keywords)
        return f"({clothing_keywords}:{self.prompt_weight})"
    
    def select_appropriate_value(self, context: Dict[str, Any]) -> str:
        """Select appropriate clothing style value."""
        self._selected_value = self.name
        return self.name
    
    def is_nude(self) -> bool:
        """Check if this clothing style represents nudity."""
        return self.name == "nude"


class AestheticStyleAttribute(SmartAttribute):
    """Intelligent aesthetic style attribute."""
    
    def __init__(self, config: AttributeConfig):
        """Initialize aesthetic style."""
        super().__init__(config)
        # Block inappropriate styles
        self._blocked_styles = {"schoolgirl"}  # Blocked for safety
    
    def generate_prompt_segment(self) -> str:
        """Generate prompt segment for aesthetic style."""
        if not self._selected_value or self.name in self._blocked_styles:
            return ""
        
        # Apply weight to aesthetic keywords
        aesthetic_keywords = ", ".join(self.config.keywords)
        return f"({aesthetic_keywords}:{self.prompt_weight})"
    
    def select_appropriate_value(self, context: Dict[str, Any]) -> str:
        """Select appropriate aesthetic style value."""
        # Ensure we don't select blocked styles
        if self.name in self._blocked_styles:
            # Select a different, appropriate style
            self._selected_value = "goth" if self.name == "schoolgirl" else self.name
        else:
            self._selected_value = self.name
        
        return self._selected_value


class CharacterAttributeManager:
    """Manager for coordinating intelligent character attributes."""
    
    def __init__(self):
        """Initialize attribute manager."""
        self.attributes: Dict[AttributeCategory, List[SmartAttribute]] = {}
        self.selected_attributes: Dict[AttributeCategory, SmartAttribute] = {}
    
    def register_attribute(self, attribute: SmartAttribute):
        """
        Register an attribute with the manager.
        
        Args:
            attribute: Attribute to register
        """
        category = attribute.category
        if category not in self.attributes:
            self.attributes[category] = []
        
        self.attributes[category].append(attribute)
    
    def select_attribute(self, category: AttributeCategory, 
                         context: Dict[str, Any] = None) -> SmartAttribute:
        """
        Select an attribute from a category.
        
        Args:
            category: Category to select from
            context: Selection context
            
        Returns:
            Selected attribute
        """
        if context is None:
            context = {}
        
        # Get available attributes in this category
        available_attributes = self.attributes.get(category, [])
        if not available_attributes:
            raise ValueError(f"No attributes available for category {category}")
        
        # Filter by age if provided
        age = context.get("age")
        if age:
            available_attributes = [
                attr for attr in available_attributes 
                if attr.validate_age(age)
            ]
        
        # Filter by compatibility with already selected attributes
        compatible_attributes = []
        for attr in available_attributes:
            is_compatible = True
            for selected_attr in self.selected_attributes.values():
                if not attr.is_compatible_with(selected_attr):
                    is_compatible = False
                    break
            
            if is_compatible:
                compatible_attributes.append(attr)
        
        # If no compatible attributes, relax compatibility rules
        if not compatible_attributes:
            compatible_attributes = available_attributes
        
        # Weighted selection based on probabilities
        if compatible_attributes:
            weights = [attr.probability for attr in compatible_attributes]
            selected_attribute = self._weighted_choice(compatible_attributes, weights)
        else:
            # Fallback: select any available attribute
            selected_attribute = available_attributes[0]
        
        # Store selected attribute
        self.selected_attributes[category] = selected_attribute
        
        # Associate with other selected attributes
        for other_attr in self.selected_attributes.values():
            if other_attr != selected_attribute:
                selected_attribute.associate_attribute(other_attr)
                other_attr.associate_attribute(selected_attribute)
        
        # Perform value selection
        selected_attribute.select_appropriate_value(context)
        
        return selected_attribute
    
    def _weighted_choice(self, items: List[SmartAttribute], 
                        weights: List[float]) -> SmartAttribute:
        """
        Select item based on weights.
        
        Args:
            items: Items to select from
            weights: Corresponding weights
            
        Returns:
            Selected item
        """
        if not items:
            raise ValueError("No items to choose from")
        
        if len(items) != len(weights):
            raise ValueError("Items and weights must have same length")
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            # Equal probability for all
            return items[0]  # Or random choice
        
        probabilities = [w / total_weight for w in weights]
        
        # Random selection
        import random
        return random.choices(items, weights=probabilities, k=1)[0]
    
    def generate_complete_prompt(self) -> str:
        """
        Generate complete prompt from all selected attributes.
        
        Returns:
            Complete formatted prompt
        """
        segments = []
        
        # Generate segments in logical order
        ordered_categories = [
            AttributeCategory.AGE,
            AttributeCategory.ETHNICITY,
            AttributeCategory.SKIN_TONE,
            AttributeCategory.EYE_COLOR,
            AttributeCategory.HAIR_COLOR,
            AttributeCategory.HAIR_TEXTURE,
            AttributeCategory.BODY_TYPE,
            AttributeCategory.BREAST_SIZE,
            AttributeCategory.PHYSICAL_FEATURES,
            AttributeCategory.BODY_SIZES,
            AttributeCategory.AESTHETIC_STYLES,
            AttributeCategory.FANTASY_RACES,
            AttributeCategory.CLOTHING_STYLES,
            AttributeCategory.CLOTHING_CONDITIONS,
            AttributeCategory.CLOTHING_DETAILS,
            AttributeCategory.ACCESSORIES,
            AttributeCategory.EMOTIONAL_STATES,
            AttributeCategory.POSITIONS,
            AttributeCategory.ENVIRONMENTS,
            AttributeCategory.SETTINGS,
            AttributeCategory.COSPLAY_STYLES,
            AttributeCategory.ERATIC_TOYS,
            AttributeCategory.WEATHER,
            AttributeCategory.SPECIAL_EFFECTS,
            AttributeCategory.ARTISTIC_STYLES,
        ]
        
        for category in ordered_categories:
            attr = self.selected_attributes.get(category)
            if attr:
                segment = attr.generate_prompt_segment()
                if segment:
                    segments.append(segment)
        
        return ", ".join(segments)
    
    def validate_selections(self) -> List[Tuple[SmartAttribute, SmartAttribute, str]]:
        """
        Validate all selected attribute combinations.
        
        Returns:
            List of validation issues (attr1, attr2, issue_description)
        """
        issues = []
        
        # Check all pairs of selected attributes
        selected_list = list(self.selected_attributes.values())
        
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                attr_a = selected_list[i]
                attr_b = selected_list[j]
                
                if not attr_a.is_compatible_with(attr_b):
                    issues.append((
                        attr_a, 
                        attr_b, 
                        f"Incompatible attributes: {attr_a.name} and {attr_b.name}"
                    ))
        
        return issues