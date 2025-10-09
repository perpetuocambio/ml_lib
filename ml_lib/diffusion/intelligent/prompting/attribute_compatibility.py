"""Attribute compatibility rules for coherent character generation."""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass


@dataclass
class CompatibilityRule:
    """Defines compatibility rules between attribute categories."""
    
    # Categories that must be compatible
    category_a: str
    category_b: str
    
    # Compatible combinations (None means all are compatible)
    compatible_combinations: List[Tuple[str, str]] = None
    
    # Incompatible combinations
    incompatible_combinations: List[Tuple[str, str]] = None
    
    # Required combinations (both must appear together)
    required_combinations: List[Tuple[str, str]] = None


class AttributeCompatibilityChecker:
    """Checks compatibility between different character attributes."""
    
    def __init__(self):
        """Initialize compatibility rules."""
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> List[CompatibilityRule]:
        """Initialize all compatibility rules."""
        rules = [
            # Age-related restrictions
            CompatibilityRule(
                category_a="age_ranges",
                category_b="aesthetic_styles",
                incompatible_combinations=[
                    # Schoolgirl/Student aesthetics only for certain age ranges
                    ("young_adult", "schoolgirl"),  # Removed due to inappropriateness
                    ("adult", "schoolgirl"),  # Removed due to inappropriateness
                ]
            ),
            
            # Clothing and nudity compatibility
            CompatibilityRule(
                category_a="clothing_styles",
                category_b="activities",
                incompatible_combinations=[
                    # Formal clothing with explicit activities
                    ("formal", "explicit_sexual"),
                    ("formal", "bdsm"),
                ],
                required_combinations=[
                    # Nude requires explicit activities
                    ("nude", "explicit_sexual"),
                    ("nude", "intimate"),
                ]
            ),
            
            # Fantasy races and realistic styles
            CompatibilityRule(
                category_a="fantasy_races",
                category_b="artistic_styles",
                incompatible_combinations=[
                    # Realistic styles with fantasy races
                    ("elf", "photorealistic"),
                    ("demon", "photorealistic"),
                ]
            ),
            
            # Cosplay and clothing styles
            CompatibilityRule(
                category_a="cosplay_styles",
                category_b="clothing_styles",
                incompatible_combinations=[
                    # Cosplay overrides normal clothing
                    ("anime", "formal"),
                    ("fantasy", "casual"),
                ]
            ),
            
            # Body size and clothing
            CompatibilityRule(
                category_a="body_sizes",
                category_b="clothing_styles",
                incompatible_combinations=[
                    # Pregnancy with revealing clothing
                    ("pregnant", "lingerie"),
                ]
            ),
            
            # Age features and clothing
            CompatibilityRule(
                category_a="age_ranges",
                category_b="clothing_styles",
                incompatible_combinations=[
                    # Very young looks with mature clothing
                    ("young_adult", "mature"),
                ]
            )
        ]
        
        return rules
    
    def check_compatibility(self, attribute_a: Tuple[str, str], attribute_b: Tuple[str, str]) -> bool:
        """
        Check if two attributes are compatible.
        
        Args:
            attribute_a: (category, value) tuple
            attribute_b: (category, value) tuple
            
        Returns:
            True if compatible, False otherwise
        """
        category_a, value_a = attribute_a
        category_b, value_b = attribute_b
        
        # Check all rules
        for rule in self.rules:
            # Check if this rule applies to these categories
            if ((rule.category_a == category_a and rule.category_b == category_b) or
                (rule.category_a == category_b and rule.category_b == category_a)):
                
                # Check incompatible combinations
                if rule.incompatible_combinations:
                    for inc_a, inc_b in rule.incompatible_combinations:
                        if ((value_a == inc_a and value_b == inc_b) or 
                            (value_a == inc_b and value_b == inc_a)):
                            return False
                
                # Check required combinations (if specified, both must be compatible or none)
                if rule.required_combinations:
                    has_required = False
                    for req_a, req_b in rule.required_combinations:
                        if ((value_a == req_a and value_b == req_b) or 
                            (value_a == req_b and value_b == req_a)):
                            has_required = True
                            break
                    
                    # If there are required combinations and this pair doesn't match any,
                    # we need to check if it violates any rule
                    if not has_required:
                        # This is handled by incompatible combinations check above
                        
                # Check compatible combinations (if specified)
                if (rule.compatible_combinations and 
                    rule.compatible_combinations != [None]):
                    is_compatible = False
                    for comp_a, comp_b in rule.compatible_combinations:
                        if ((value_a == comp_a and value_b == comp_b) or 
                            (value_a == comp_b and value_b == comp_a)):
                            is_compatible = True
                            break
                    
                    # If specific combinations are defined and this isn't one of them, it's incompatible
                    if not is_compatible:
                        return False
        
        # If no rules say it's incompatible, it's compatible
        return True
    
    def get_incompatible_attributes(self, selected_attributes: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Get list of incompatible attribute pairs from selected attributes.
        
        Args:
            selected_attributes: List of (category, value) tuples
            
        Returns:
            List of incompatible pairs
        """
        incompatible_pairs = []
        
        # Check all pairs
        for i in range(len(selected_attributes)):
            for j in range(i + 1, len(selected_attributes)):
                attr_a = selected_attributes[i]
                attr_b = selected_attributes[j]
                
                if not self.check_compatibility(attr_a, attr_b):
                    incompatible_pairs.append((attr_a, attr_b))
        
        return incompatible_pairs
    
    def suggest_compatible_combinations(self, selected_attribute: Tuple[str, str], 
                                      target_categories: List[str]) -> Dict[str, List[str]]:
        """
        Suggest compatible attributes for a given attribute.
        
        Args:
            selected_attribute: (category, value) tuple
            target_categories: Categories to suggest compatible attributes for
            
        Returns:
            Dictionary mapping category to list of compatible values
        """
        suggestions = {}
        category_a, value_a = selected_attribute
        
        # For each target category, find compatible values
        for category_b in target_categories:
            compatible_values = []
            
            # Check all rules that involve these categories
            for rule in self.rules:
                if ((rule.category_a == category_a and rule.category_b == category_b) or
                    (rule.category_a == category_b and rule.category_b == category_a)):
                    
                    # If incompatible combinations are specified, avoid those
                    if rule.incompatible_combinations:
                        # Get all possible values for this category (this would need to be passed in)
                        # For now, we'll just note which combinations are incompatible
                        pass
                    
                    # If compatible combinations are specified, use those
                    if (rule.compatible_combinations and 
                        rule.compatible_combinations != [None]):
                        for comp_a, comp_b in rule.compatible_combinations:
                            if rule.category_a == category_a and rule.category_b == category_b:
                                if comp_a == value_a:
                                    compatible_values.append(comp_b)
                            elif rule.category_a == category_b and rule.category_b == category_a:
                                if comp_b == value_a:
                                    compatible_values.append(comp_a)
            
            suggestions[category_b] = compatible_values
        
        return suggestions