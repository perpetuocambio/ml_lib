"""Attribute groups and combinations for coherent character generation."""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from ml_lib.diffusion.intelligent.prompting.smart_attributes import (
    SmartAttribute, AttributeCategory, AttributeConfig
)


class AttributeGroupType(Enum):
    """Types of attribute groups."""
    
    # Age-related combinations
    MATURITY_SET = "maturity_set"  # Mature features, older age, etc.
    YOUTHFUL_SET = "youthful_set"   # Youthful appearance, younger age, etc.
    
    # Style combinations
    GOTH_SET = "goth_set"          # Goth aesthetic, dark clothing, etc.
    PUNK_SET = "punk_set"          # Punk style, rebellious look, etc.
    NURSE_SET = "nurse_set"        # Nurse outfit, medical theme, etc.
    FETISH_SET = "fetish_set"      # Fetish wear, latex, leather, etc.
    
    # Fantasy combinations
    ELF_SET = "elf_set"            # Elf features, fantasy aesthetic, etc.
    DEMON_SET = "demon_set"        # Demon characteristics, dark themes, etc.
    
    # Seasonal combinations
    SUMMER_SET = "summer_set"      # Beachwear, sunny themes, etc.
    WINTER_SET = "winter_set"      # Winter clothing, cold themes, etc.
    
    # Emotional combinations
    SENSUAL_SET = "sensual_set"    # Sensual expressions, seductive look, etc.
    INTENSE_SET = "intense_set"    # Intense emotions, dramatic expressions, etc.


@dataclass
class AttributeGroup:
    """A group of attributes that work well together."""
    
    # Group identification
    name: str
    group_type: AttributeGroupType
    
    # Attributes in this group
    attributes: Dict[AttributeCategory, str] = field(default_factory=dict)
    
    # Probability of selecting this group
    probability: float = 1.0
    
    # Age range for this group
    min_age: int = 18
    max_age: int = 80
    
    # Keywords that define this group
    defining_keywords: List[str] = field(default_factory=list)
    
    # Complementary groups (groups that work well with this one)
    complementary_groups: List[AttributeGroupType] = field(default_factory=list)
    
    # Conflicting groups (groups that shouldn't be combined with this one)
    conflicting_groups: List[AttributeGroupType] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttributeGroupManager:
    """Manages groups of attributes that work well together."""
    
    def __init__(self):
        """Initialize group manager."""
        self.groups: List[AttributeGroup] = []
        self._group_lookup: Dict[AttributeGroupType, AttributeGroup] = {}
    
    def register_group(self, group: AttributeGroup):
        """
        Register an attribute group.
        
        Args:
            group: Group to register
        """
        self.groups.append(group)
        self._group_lookup[group.group_type] = group
    
    def get_group(self, group_type: AttributeGroupType) -> Optional[AttributeGroup]:
        """
        Get a registered group by type.
        
        Args:
            group_type: Type of group to get
            
        Returns:
            Group or None if not found
        """
        return self._group_lookup.get(group_type)
    
    def get_compatible_groups(self, group_type: AttributeGroupType) -> List[AttributeGroup]:
        """
        Get groups that are compatible with a given group.
        
        Args:
            group_type: Base group type
            
        Returns:
            List of compatible groups
        """
        base_group = self.get_group(group_type)
        if not base_group:
            return []
        
        compatible = []
        for group in self.groups:
            if group.group_type not in base_group.conflicting_groups:
                compatible.append(group)
        
        return compatible
    
    def get_conflicting_groups(self, group_type: AttributeGroupType) -> List[AttributeGroup]:
        """
        Get groups that conflict with a given group.
        
        Args:
            group_type: Base group type
            
        Returns:
            List of conflicting groups
        """
        base_group = self.get_group(group_type)
        if not base_group:
            return []
        
        conflicting = []
        for conflict_type in base_group.conflicting_groups:
            conflict_group = self.get_group(conflict_type)
            if conflict_group:
                conflicting.append(conflict_group)
        
        return conflicting
    
    def select_group_for_age(self, age: int) -> Optional[AttributeGroup]:
        """
        Select an appropriate group for a given age.
        
        Args:
            age: Target age
            
        Returns:
            Selected group or None
        """
        # Filter groups by age
        age_appropriate_groups = [
            group for group in self.groups
            if group.min_age <= age <= group.max_age
        ]
        
        if not age_appropriate_groups:
            return None
        
        # Weighted selection based on probabilities
        weights = [group.probability for group in age_appropriate_groups]
        total_weight = sum(weights)
        
        if total_weight > 0:
            import random
            normalized_weights = [w / total_weight for w in weights]
            return random.choices(age_appropriate_groups, weights=normalized_weights, k=1)[0]
        else:
            # Equal probability
            import random
            return random.choice(age_appropriate_groups)
    
    def validate_group_combination(self, groups: List[AttributeGroup]) -> bool:
        """
        Validate that a combination of groups is compatible.
        
        Args:
            groups: Groups to validate
            
        Returns:
            True if compatible, False otherwise
        """
        # Check for direct conflicts
        group_types = {group.group_type for group in groups}
        
        for group in groups:
            for conflict_type in group.conflicting_groups:
                if conflict_type in group_types:
                    return False  # Direct conflict found
        
        return True  # No conflicts found


# Pre-defined attribute groups
def create_standard_groups() -> AttributeGroupManager:
    """Create standard attribute groups."""
    
    manager = AttributeGroupManager()
    
    # Maturity group (40+ year old women)
    maturity_group = AttributeGroup(
        name="Mature Woman Characteristics",
        group_type=AttributeGroupType.MATURITY_SET,
        attributes={
            AttributeCategory.AGE: "milf",
            AttributeCategory.BODY_TYPE: "curvy",
            AttributeCategory.SKIN_TONE: "medium",
            AttributeCategory.HAIR_TEXTURE: "wavy",
            AttributeCategory.EMOTIONAL_STATES: "sensual",
        },
        probability=0.4,
        min_age=40,
        max_age=80,
        defining_keywords=["mature", "experienced", "confident", "wisdom"],
        complementary_groups=[AttributeGroupType.SENSUAL_SET],
        conflicting_groups=[AttributeGroupType.YOUTHFUL_SET]
    )
    manager.register_group(maturity_group)
    
    # Youthful group (18-35 year old women)
    youthful_group = AttributeGroup(
        name="Young Adult Characteristics",
        group_type=AttributeGroupType.YOUTHFUL_SET,
        attributes={
            AttributeCategory.AGE: "young_adult",
            AttributeCategory.BODY_TYPE: "slim",
            AttributeCategory.SKIN_TONE: "light",
            AttributeCategory.HAIR_TEXTURE: "straight",
            AttributeCategory.EMOTIONAL_STATES: "happy",
        },
        probability=0.3,
        min_age=18,
        max_age=35,
        defining_keywords=["youthful", "energetic", "fresh", "vibrant"],
        complementary_groups=[AttributeGroupType.SUMMER_SET],
        conflicting_groups=[AttributeGroupType.MATURITY_SET]
    )
    manager.register_group(youthful_group)
    
    # Goth group
    goth_group = AttributeGroup(
        name="Gothic Style",
        group_type=AttributeGroupType.GOTH_SET,
        attributes={
            AttributeCategory.AESTHETIC_STYLES: "goth",
            AttributeCategory.CLOTHING_STYLES: "fetish",
            AttributeCategory.ACCESSORIES: "jewelry",
            AttributeCategory.COLOR_SCHEMES: "dark",
        },
        probability=0.05,
        min_age=18,
        max_age=80,
        defining_keywords=["dark", "gothic", "mysterious", "elegant"],
        complementary_groups=[AttributeGroupType.DEMON_SET],
        conflicting_groups=[]
    )
    manager.register_group(goth_group)
    
    # Nurse group
    nurse_group = AttributeGroup(
        name="Nurse Outfit",
        group_type=AttributeGroupType.NURSE_SET,
        attributes={
            AttributeCategory.AESTHETIC_STYLES: "nurse",
            AttributeCategory.CLOTHING_STYLES: "fetish",
            AttributeCategory.ACCESSORIES: "headwear",
            AttributeCategory.SETTINGS: "hospital",
        },
        probability=0.1,
        min_age=18,
        max_age=80,
        defining_keywords=["medical", "nurse", "professional", "caregiver"],
        complementary_groups=[],
        conflicting_groups=[AttributeGroupType.SCHOOLGIRL_SET]  # Explicitly blocked
    )
    manager.register_group(nurse_group)
    
    # Fetish group
    fetish_group = AttributeGroup(
        name="Fetish Wear",
        group_type=AttributeGroupType.FETISH_SET,
        attributes={
            AttributeCategory.CLOTHING_STYLES: "fetish",
            AttributeCategory.ACCESSORIES: "fetish_accessories",
            AttributeCategory.ERATIC_TOYS: "bdsm",
        },
        probability=0.15,
        min_age=18,
        max_age=80,
        defining_keywords=["fetish", "bondage", "latex", "leather"],
        complementary_groups=[],
        conflicting_groups=[AttributeGroupType.SCHOOLGIRL_SET]  # Explicitly blocked
    )
    manager.register_group(fetish_group)
    
    # Elf group (fantasy)
    elf_group = AttributeGroup(
        name="Elf Characteristics",
        group_type=AttributeGroupType.ELF_SET,
        attributes={
            AttributeCategory.FANTASY_RACES: "elf",
            AttributeCategory.HAIR_TEXTURE: "straight",
            AttributeCategory.EMOTIONAL_STATES: "neutral",
            AttributeCategory.ARTISTIC_STYLES: "fantasy",
        },
        probability=0.05,
        min_age=18,
        max_age=80,
        defining_keywords=["elf", "magical", "ethereal", "graceful"],
        complementary_groups=[AttributeGroupType.FANTASY_SET],
        conflicting_groups=[AttributeGroupType.MATURE_SET]  # Age/ethnicity inconsistencies
    )
    manager.register_group(elf_group)
    
    # Demon group (fantasy)
    demon_group = AttributeGroup(
        name="Demon Characteristics",
        group_type=AttributeGroupType.DEMON_SET,
        attributes={
            AttributeCategory.FANTASY_RACES: "demon",
            AttributeCategory.HAIR_COLOR: "red",
            AttributeCategory.EYE_COLOR: "red",
            AttributeCategory.SKIN_TONE: "dark",
            AttributeCategory.ARTISTIC_STYLES: "fantasy",
        },
        probability=0.05,
        min_age=18,
        max_age=80,
        defining_keywords=["demon", "supernatural", "dark", "powerful"],
        complementary_groups=[AttributeGroupType.GOTH_SET],
        conflicting_groups=[AttributeGroupType.YOUTHFUL_SET]  # Demons are typically not portrayed as young
    )
    manager.register_group(demon_group)
    
    # Sensual group
    sensual_group = AttributeGroup(
        name="Sensual Expression",
        group_type=AttributeGroupType.SENSUAL_SET,
        attributes={
            AttributeCategory.EMOTIONAL_STATES: "sensual",
            AttributeCategory.BODY_LANGUAGE: "seductive",
            AttributeCategory.EXPRESSIONS: "inviting",
        },
        probability=0.2,
        min_age=18,
        max_age=80,
        defining_keywords=["sensual", "seductive", "inviting", "alluring"],
        complementary_groups=[AttributeGroupType.MATURITY_SET],
        conflicting_groups=[]
    )
    manager.register_group(sensual_group)
    
    # Intense group
    intense_group = AttributeGroup(
        name="Intense Expression",
        group_type=AttributeGroupType.INTENSE_SET,
        attributes={
            AttributeCategory.EMOTIONAL_STATES: "intense",
            AttributeCategory.BODY_LANGUAGE: "passionate",
            AttributeCategory.EXPRESSIONS: "focused",
        },
        probability=0.15,
        min_age=18,
        max_age=80,
        defining_keywords=["intense", "passionate", "focused", "powerful"],
        complementary_groups=[AttributeGroupType.FETISH_SET],
        conflicting_groups=[]
    )
    manager.register_group(intense_group)
    
    return manager


# Relationship rules between attributes
class AttributeRelationships:
    """Defines relationships and dependencies between attributes."""
    
    def __init__(self):
        """Initialize relationship rules."""
        self.relationships = self._define_relationships()
    
    def _define_relationships(self) -> Dict[str, Dict[str, float]]:
        """
        Define relationship strengths between attribute categories.
        
        Returns:
            Dictionary mapping (category_a, category_b) to relationship strength (0-1)
        """
        relationships = {
            # Strong relationships
            ("age_ranges", "body_types"): 0.9,
            ("ethnicities", "skin_tones"): 0.95,
            ("hair_colors", "hair_textures"): 0.8,
            ("aesthetic_styles", "clothing_styles"): 0.85,
            ("fantasy_races", "artistic_styles"): 0.7,
            
            # Moderate relationships
            ("body_types", "breast_sizes"): 0.6,
            ("emotional_states", "body_language"): 0.7,
            ("weather_conditions", "environments"): 0.65,
            ("cosplay_styles", "accessories"): 0.6,
            
            # Weak relationships
            ("eye_colors", "hair_colors"): 0.3,
            ("accessories", "special_effects"): 0.25,
            
            # Negative relationships (conflicts)
            ("schoolgirl", "nude"): -0.5,  # Explicitly blocked
            ("nun", "explicit_sexual"): -1.0,  # Completely incompatible
            ("pregnant", "revealing_clothing"): -0.7,  # Generally incompatible
        }
        
        return relationships
    
    def get_relationship_strength(self, category_a: str, category_b: str) -> float:
        """
        Get relationship strength between two categories.
        
        Args:
            category_a: First category
            category_b: Second category
            
        Returns:
            Relationship strength (-1 to 1, where negative means conflict)
        """
        # Check direct relationship
        strength = self.relationships.get((category_a, category_b))
        if strength is not None:
            return strength
        
        # Check reverse relationship
        strength = self.relationships.get((category_b, category_a))
        if strength is not None:
            return strength
        
        # No defined relationship
        return 0.0
    
    def are_directly_conflicting(self, category_a: str, category_b: str) -> bool:
        """
        Check if two categories are directly conflicting.
        
        Args:
            category_a: First category
            category_b: Second category
            
        Returns:
            True if directly conflicting
        """
        strength = self.get_relationship_strength(category_a, category_b)
        return strength < 0
    
    def are_strongly_related(self, category_a: str, category_b: str) -> bool:
        """
        Check if two categories are strongly related.
        
        Args:
            category_a: First category
            category_b: Second category
            
        Returns:
            True if strongly related (strength > 0.7)
        """
        strength = self.get_relationship_strength(category_a, category_b)
        return strength > 0.7


# Compatibility checker
class CompatibilityChecker:
    """Checks compatibility between selected attributes."""
    
    def __init__(self):
        """Initialize compatibility checker."""
        self.relationships = AttributeRelationships()
        self.group_manager = create_standard_groups()
    
    def check_compatibility(self, selected_attributes: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Check if selected attributes are compatible.
        
        Args:
            selected_attributes: Dictionary mapping categories to selected values
            
        Returns:
            Tuple of (is_compatible, list_of_issues)
        """
        issues = []
        
        # Check for blocked combinations
        blocked_combinations = [
            ("aesthetic_styles", "schoolgirl"),  # Explicitly blocked
            ("activities", "underage"),  # Explicitly blocked
            ("clothing_styles", "school_uniform"),  # Blocked variant
        ]
        
        for category, value in selected_attributes.items():
            # Check if any selected value is blocked
            for blocked_cat, blocked_val in blocked_combinations:
                if category == blocked_cat and value == blocked_val:
                    issues.append(f"Blocked attribute: {category} = {value}")
        
        # Check for direct conflicts between attributes
        categories = list(selected_attributes.keys())
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                cat_a = categories[i]
                cat_b = categories[j]
                
                if self.relationships.are_directly_conflicting(cat_a, cat_b):
                    val_a = selected_attributes[cat_a]
                    val_b = selected_attributes[cat_b]
                    issues.append(f"Conflicting attributes: {cat_a}={val_a} and {cat_b}={val_b}")
        
        # Check group compatibility
        selected_groups = []
        for attr_category, attr_value in selected_attributes.items():
            # This would require mapping attributes to groups
            # For now, we'll skip detailed group checking
            
            # Simple check: if we have "schoolgirl" aesthetic, flag it
            if attr_category == "aesthetic_styles" and attr_value == "schoolgirl":
                issues.append("Schoolgirl aesthetic is not permitted")
        
        is_compatible = len(issues) == 0
        return is_compatible, issues