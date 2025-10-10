"""Compatibility map type for attribute compatibility tracking."""

from dataclasses import dataclass, field
from typing import Dict, List

from ml_lib.diffusion.intelligent.prompting.models.attribute_type import AttributeType
from ml_lib.diffusion.intelligent.prompting.models.attribute_definition import AttributeDefinition


@dataclass
class CompatibilityMap:
    """Map of compatible attributes by type.

    Replaces Dict[AttributeType, List[AttributeDefinition]] with strongly typed class.
    """

    compatible_by_type: Dict[AttributeType, List[AttributeDefinition]] = field(default_factory=dict)

    def add_compatible_attributes(
        self,
        attribute_type: AttributeType,
        attributes: List[AttributeDefinition]
    ) -> None:
        """Add compatible attributes for a type.

        Args:
            attribute_type: Type of attributes
            attributes: List of compatible attributes
        """
        self.compatible_by_type[attribute_type] = attributes

    def get_compatible_attributes(
        self,
        attribute_type: AttributeType
    ) -> List[AttributeDefinition]:
        """Get compatible attributes for a type.

        Args:
            attribute_type: Type of attributes to get

        Returns:
            List of compatible attributes (empty if none found)
        """
        return self.compatible_by_type.get(attribute_type, [])

    def has_compatible_attributes(self, attribute_type: AttributeType) -> bool:
        """Check if there are compatible attributes for a type.

        Args:
            attribute_type: Type to check

        Returns:
            True if compatible attributes exist
        """
        return attribute_type in self.compatible_by_type and len(self.compatible_by_type[attribute_type]) > 0

    @property
    def total_compatible_count(self) -> int:
        """Total number of compatible attributes across all types."""
        return sum(len(attrs) for attrs in self.compatible_by_type.values())

    @property
    def compatible_types(self) -> List[AttributeType]:
        """List of attribute types that have compatible attributes."""
        return list(self.compatible_by_type.keys())
