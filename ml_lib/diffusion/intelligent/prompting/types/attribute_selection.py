"""Attribute selection type for character generation."""

from dataclasses import dataclass, field
from typing import Dict, Optional

from ml_lib.diffusion.intelligent.prompting.models.attribute_type import AttributeType


@dataclass
class AttributeSelection:
    """Selection of attributes by type for character generation.

    Replaces Dict[str, str] returns with strongly typed class.
    """

    selected_attributes: Dict[AttributeType, str] = field(default_factory=dict)

    def set_attribute(self, attribute_type: AttributeType, value: str) -> None:
        """Set an attribute selection.

        Args:
            attribute_type: Type of attribute
            value: Selected value
        """
        self.selected_attributes[attribute_type] = value

    def get_attribute(self, attribute_type: AttributeType) -> Optional[str]:
        """Get selected attribute value.

        Args:
            attribute_type: Type of attribute

        Returns:
            Selected value or None if not set
        """
        return self.selected_attributes.get(attribute_type)

    def has_attribute(self, attribute_type: AttributeType) -> bool:
        """Check if an attribute is selected.

        Args:
            attribute_type: Type of attribute

        Returns:
            True if attribute is selected
        """
        return attribute_type in self.selected_attributes

    @property
    def age_range(self) -> Optional[str]:
        """Get selected age range."""
        return self.get_attribute(AttributeType.AGE_RANGE)

    @property
    def ethnicity(self) -> Optional[str]:
        """Get selected ethnicity."""
        return self.get_attribute(AttributeType.ETHNICITY)

    @property
    def skin_tone(self) -> Optional[str]:
        """Get selected skin tone."""
        return self.get_attribute(AttributeType.SKIN_TONE)

    @property
    def body_type(self) -> Optional[str]:
        """Get selected body type."""
        return self.get_attribute(AttributeType.BODY_TYPE)

    @property
    def clothing_style(self) -> Optional[str]:
        """Get selected clothing style."""
        return self.get_attribute(AttributeType.CLOTHING_STYLE)

    @property
    def aesthetic_style(self) -> Optional[str]:
        """Get selected aesthetic style."""
        return self.get_attribute(AttributeType.AESTHETIC_STYLE)

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for backward compatibility.

        Returns:
            Dictionary mapping attribute type values to selections
        """
        return {attr_type.value: value for attr_type, value in self.selected_attributes.items()}
