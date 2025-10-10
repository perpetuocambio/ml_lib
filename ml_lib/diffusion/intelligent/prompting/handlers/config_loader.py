"""Enhanced configuration loader using class-based approach."""

import yaml
from pathlib import Path
from typing import List, Optional

from ml_lib.diffusion.intelligent.prompting.core import AttributeType, AttributeDefinition
from ml_lib.diffusion.intelligent.prompting.handlers import CharacterAttributeSet
from ml_lib.diffusion.intelligent.prompting.models import ValidationResult, CompatibilityMap


class ConfigLoader:
    """Configuration loader that converts YAML to class-based attributes."""

    # Safety terms for content filtering
    BLOCKED_TERMS = ['schoolgirl', 'school uniform', 'underage', 'minor', 'child', 'teen']

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize enhanced config loader.
        
        Args:
            config_dir: Directory containing YAML config files (defaults to project config)
        """
        if config_dir is None:
            # Default to project config directory
            module_dir = Path(__file__).parent
            project_root = module_dir.parent.parent.parent.parent
            config_dir = project_root / "config" / "intelligent_prompting"
        
        self.config_dir = config_dir
        self.attribute_set = CharacterAttributeSet()
        
        # Load configuration
        self._load_all_configurations()
    
    def _load_all_configurations(self):
        """Load all configuration files."""
        # Load main character attributes
        character_config_path = self.config_dir / "character_attributes.yaml"
        if character_config_path.exists():
            self.attribute_set.load_from_yaml(character_config_path)
        
        # Load other configuration files if they exist
        # (This could be expanded to load concept categories, etc.)
    
    def get_attribute_set(self) -> CharacterAttributeSet:
        """
        Get the loaded attribute set.
        
        Returns:
            Character attribute set with all loaded attributes
        """
        return self.attribute_set
    
    def get_attribute(self, attribute_type: AttributeType, name: str) -> Optional[AttributeDefinition]:
        """
        Get a specific attribute.
        
        Args:
            attribute_type: Type of attribute
            name: Name of attribute
            
        Returns:
            Attribute definition or None if not found
        """
        return self.attribute_set.get_attribute(attribute_type, name)
    
    def get_compatible_attributes(self, selected_attributes: List[AttributeDefinition]) -> CompatibilityMap:
        """
        Get attributes that are compatible with currently selected attributes.

        Args:
            selected_attributes: Currently selected attributes

        Returns:
            CompatibilityMap with compatible attributes by type
        """
        compatibility_map = CompatibilityMap()

        for attr_type in AttributeType:
            collection = self.attribute_set.get_collection(attr_type)
            if collection:
                compatible_attributes = collection.get_compatible_attributes(selected_attributes)
                if compatible_attributes:
                    compatibility_map.add_compatible_attributes(attr_type, compatible_attributes)

        return compatibility_map
    
    def validate_character_selection(self, selected_attributes: List[AttributeDefinition]) -> ValidationResult:
        """
        Validate a complete character selection.

        Args:
            selected_attributes: All selected attributes for a character

        Returns:
            ValidationResult with compatibility, issues, and suggestions
        """
        # Validate compatibility
        is_compatible, compatibility_issues = self.attribute_set.validate_compatibility(selected_attributes)

        # Validate age consistency
        age_consistency_issues = self._validate_age_consistency(selected_attributes)

        # Check for blocked content
        blocked_content_issues = self._check_for_blocked_content(selected_attributes)

        # Collect all issues
        all_issues = compatibility_issues + age_consistency_issues + blocked_content_issues

        # Generate suggestions for improvement
        suggestions = self._generate_improvement_suggestions(selected_attributes, all_issues)

        return ValidationResult(
            is_valid=len(all_issues) == 0,
            compatibility_valid=is_compatible,
            issues=all_issues,
            age_consistency_issues=age_consistency_issues,
            blocked_content_issues=blocked_content_issues,
            suggestions=suggestions
        )
    
    def _validate_age_consistency(self, selected_attributes: List[AttributeDefinition]) -> List[str]:
        """
        Validate age consistency among selected attributes.
        
        Args:
            selected_attributes: Selected attributes to validate
            
        Returns:
            List of age consistency issues
        """
        issues = []
        
        # Find age-related attributes
        age_attributes = [
            attr for attr in selected_attributes 
            if attr.attribute_type == AttributeType.AGE_RANGE
        ]
        
        # Find attributes with age restrictions
        restricted_attributes = [
            attr for attr in selected_attributes 
            if attr.min_age > 18 or attr.max_age < 80
        ]
        
        # Check if restricted attributes are compatible with age attributes
        for age_attr in age_attributes:
            for restricted_attr in restricted_attributes:
                if not restricted_attr.validate_age((age_attr.min_age + age_attr.max_age) // 2):
                    issues.append(
                        f"Age inconsistency: '{restricted_attr.name}' is not appropriate for age range '{age_attr.name}'"
                    )
        
        return issues
    
    def _check_for_blocked_content(self, selected_attributes: List[AttributeDefinition]) -> List[str]:
        """
        Check for blocked content in selected attributes.
        
        Args:
            selected_attributes: Selected attributes to check
            
        Returns:
            List of blocked content issues
        """
        issues = []

        for attribute in selected_attributes:
            if attribute.is_blocked:
                issues.append(f"Blocked content: '{attribute.name}' ({attribute.attribute_type.value})")

            # Additional check for keywords
            for keyword in attribute.keywords:
                if any(blocked_term in keyword.lower() for blocked_term in self.BLOCKED_TERMS):
                    issues.append(f"Potentially blocked keyword in '{attribute.name}': '{keyword}'")
        
        return issues
    
    def _generate_improvement_suggestions(self, selected_attributes: List[AttributeDefinition], 
                                        issues: List[str]) -> List[str]:
        """
        Generate suggestions for improving character selections.
        
        Args:
            selected_attributes: Currently selected attributes
            issues: Issues found during validation
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Suggest compatible alternatives for incompatible attributes
        # (This would need more sophisticated logic)
        
        # Suggest removing conflicting attributes
        for issue in issues:
            if "incompatible" in issue.lower():
                suggestions.append("Consider removing or replacing incompatible attributes to improve coherence")
        
        # Suggest age-appropriate alternatives
        for issue in issues:
            if "age inconsistency" in issue.lower():
                suggestions.append("Replace attributes that don't match the selected age range")
        
        # Suggest removing blocked content
        for issue in issues:
            if "blocked content" in issue.lower() or "potentially blocked" in issue.lower():
                suggestions.append("Remove or replace blocked content to ensure appropriate generation")
        
        # General coherence suggestions
        if len(selected_attributes) > 10:
            suggestions.append("Consider reducing the number of attributes for better focus and coherence")
        
        return suggestions


# Example usage
if __name__ == "__main__":
    # Create loader
    loader = ConfigLoader()

    # Get attribute set
    attribute_set = loader.get_attribute_set()

    # Count total attributes using all_attributes()
    total = sum(len(collection.all_attributes()) for collection in attribute_set.collections.values())
    print(f"Loaded {total} attributes")

    # Test getting specific attributes
    age_attr = loader.get_attribute(AttributeType.AGE_RANGE, "milf")
    if age_attr:
        print(f"Found age attribute: {age_attr.name} with keywords {age_attr.keywords}")

    # Test validation
    test_attributes = []
    if age_attr:
        test_attributes.append(age_attr)

    # Validate (would need more attributes for meaningful validation)
    if test_attributes:
        validation_results = loader.validate_character_selection(test_attributes)
        print(f"Validation results: {validation_results}")