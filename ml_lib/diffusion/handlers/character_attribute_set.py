"""Character attribute set handler - internal use only."""

import yaml
from pathlib import Path
from typing import List, Optional, Tuple

from ml_lib.diffusion.models import AttributeType, AttributeDefinition
from ml_lib.diffusion.handlers.attribute_collection import AttributeCollection


class CharacterAttributeSet:
    """Complete set of all character attributes organized by type."""

    def __init__(self):
        """Initialize character attribute set."""
        self.collections: dict[AttributeType, AttributeCollection] = {}

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

    def get_collection(
        self, attribute_type: AttributeType
    ) -> Optional[AttributeCollection]:
        """
        Get a collection by attribute type.

        Args:
            attribute_type: Type of collection to get

        Returns:
            Collection or None if not found
        """
        return self.collections.get(attribute_type)

    def get_attribute(
        self, attribute_type: AttributeType, name: str
    ) -> Optional[AttributeDefinition]:
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
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Process each section in the YAML
        for section_name, section_data in config.items():
            if section_name in ["randomization_rules"]:
                continue  # Skip non-attribute sections

            # Map section name to AttributeType
            try:
                attr_type = AttributeType(section_name)
            except ValueError:
                # Unknown section, skip
                continue

            # Process each attribute in the section
            for attr_name, attr_data in section_data.items():
                attribute = self._create_attribute_from_yaml(
                    attr_name, attr_type, attr_data
                )
                self.add_attribute(attribute)

    def _create_attribute_from_yaml(
        self, name: str, attr_type: AttributeType, data: dict
    ) -> AttributeDefinition:
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
        blocked_terms = [
            "schoolgirl",
            "school uniform",
            "underage",
            "minor",
            "child",
            "teen",
        ]

        # Check if this attribute should be blocked
        keywords = data.get("keywords", [])
        if any(
            blocked_term in " ".join(keywords).lower() for blocked_term in blocked_terms
        ):
            is_blocked = True

        # Create attribute
        attribute = AttributeDefinition(
            name=name,
            attribute_type=attr_type,
            keywords=keywords,
            probability=data.get("probability", 1.0),
            prompt_weight=data.get("prompt_weight", 1.0),
            min_age=data.get("min_age", 18),
            max_age=data.get("max_age", 80),
            ethnicity_associations=data.get("ethnicity_associations", []),
            lighting_suggestions=data.get("lighting_suggestions", []),
            complexity=data.get("complexity", "medium"),
            explicit=data.get("explicit", False),
            is_blocked=is_blocked,
            metadata=data,  # Store original data for reference
        )

        return attribute

    def validate_compatibility(
        self, attributes: List[AttributeDefinition]
    ) -> Tuple[bool, List[str]]:
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
                    issues.append(
                        f"Incompatible attributes: {attr_a.name} ({attr_a.attribute_type.value}) and {attr_b.name} ({attr_b.attribute_type.value})"
                    )

        # Check age consistency
        ages = [
            attr
            for attr in attributes
            if attr.attribute_type == AttributeType.AGE_RANGE
        ]
        if len(ages) > 1:
            # Check that all age attributes are consistent
            age_values = [(attr.min_age, attr.max_age) for attr in ages]
            # This could be expanded with more sophisticated age validation

        is_valid = len(issues) == 0
        return is_valid, issues
