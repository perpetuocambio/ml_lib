"""Attribute collection handler - internal use only."""

from typing import Dict, List, Optional
import random

from ml_lib.diffusion.domain.value_objects_models import AttributeType, AttributeDefinition

class AttributeCollection:
    """Collection of related attributes of the same type."""

    def __init__(self, attribute_type: AttributeType):
        """
        Initialize attribute collection.

        Args:
            attribute_type: Type of attributes in this collection
        """
        self.attribute_type = attribute_type
        # Private dict for O(1) lookup - not exposed in public API
        self._attributes: Dict[str, AttributeDefinition] = {}

    @property
    def attributes(self) -> Dict[str, AttributeDefinition]:
        """DEPRECATED: Direct access to internal dict. Use all_attributes() instead."""
        return self._attributes

    def all_attributes(self) -> List[AttributeDefinition]:
        """Get all attributes as a list (recommended over direct dict access)."""
        return list(self._attributes.values())
    
    def add_attribute(self, attribute: AttributeDefinition):
        """
        Add an attribute to the collection.

        Args:
            attribute: Attribute to add
        """
        if attribute.attribute_type != self.attribute_type:
            raise ValueError(f"Attribute type mismatch: expected {self.attribute_type}, got {attribute.attribute_type}")

        self._attributes[attribute.name] = attribute
    
    def get_attribute(self, name: str) -> Optional[AttributeDefinition]:
        """
        Get an attribute by name.

        Args:
            name: Name of attribute to get

        Returns:
            Attribute or None if not found
        """
        return self._attributes.get(name)
    
    def get_compatible_attributes(self, other_attributes: List[AttributeDefinition]) -> List[AttributeDefinition]:
        """
        Get attributes that are compatible with a list of other attributes.

        Args:
            other_attributes: List of attributes to check compatibility with

        Returns:
            List of compatible attributes
        """
        compatible = []
        for attribute in self._attributes.values():
            is_compatible = True
            for other_attr in other_attributes:
                if not attribute.is_compatible_with(other_attr):
                    is_compatible = False
                    break

            if is_compatible:
                compatible.append(attribute)

        return compatible
    
    def select_random(self) -> Optional[AttributeDefinition]:
        """
        Select an attribute with UNIFORM probability (fantasy-based, no statistical bias).

        IMPORTANT: This method uses uniform probability for all valid attributes.
        It does NOT use the probability field from YAML - all options have equal chance.

        Returns:
            Selected attribute or None if no attributes available
        """
        if not self._attributes:
            return None

        # Filter out blocked attributes
        available_attributes = [
            attr for attr in self._attributes.values()
            if not attr.is_blocked
        ]

        if not available_attributes:
            return None

        # UNIFORM random selection - all have equal probability (1/N)
        # This is intentional for fantasy content - ignore probability weights
        return random.choice(available_attributes)

    def select_by_probability(self) -> Optional[AttributeDefinition]:
        """
        DEPRECATED: Use select_random() instead.

        This method kept for backward compatibility but now uses uniform probability.
        """
        return self.select_random()


