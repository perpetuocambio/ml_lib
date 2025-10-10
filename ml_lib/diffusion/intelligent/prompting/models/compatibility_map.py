"""Compatibility map model - replaces Dict[AttributeType, List[AttributeDefinition]]."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ml_lib.diffusion.intelligent.prompting.enhanced_attributes import (
        AttributeType,
        AttributeDefinition,
    )


@dataclass
class CompatibilityMap:
    """Map of compatible attributes by type.

    This class replaces Dict[AttributeType, List[AttributeDefinition]] that was
    previously used to represent compatibility information.

    Uses a list of tuples internally for type safety, with helper methods for access.
    """

    _entries: list[tuple["AttributeType", "AttributeDefinition"]] = field(default_factory=list)

    def add(self, attr_type: "AttributeType", attribute: "AttributeDefinition") -> None:
        """Add a compatible attribute for a specific type."""
        self._entries.append((attr_type, attribute))

    def get_by_type(self, attr_type: "AttributeType") -> list["AttributeDefinition"]:
        """Get all compatible attributes for a specific type."""
        return [attr for t, attr in self._entries if t == attr_type]

    def has_type(self, attr_type: "AttributeType") -> bool:
        """Check if there are any compatible attributes for a specific type."""
        return any(t == attr_type for t, _ in self._entries)

    def all_types(self) -> set["AttributeType"]:
        """Get all attribute types that have compatible attributes."""
        return {t for t, _ in self._entries}

    def all_attributes(self) -> list["AttributeDefinition"]:
        """Get all compatible attributes across all types."""
        return [attr for _, attr in self._entries]

    @property
    def is_empty(self) -> bool:
        """Whether this compatibility map has any entries."""
        return len(self._entries) == 0

    @property
    def count(self) -> int:
        """Total number of compatible attribute entries."""
        return len(self._entries)
