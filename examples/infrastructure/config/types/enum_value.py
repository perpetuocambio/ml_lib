"""Type-safe enum value wrapper."""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class EnumValue:
    """Type-safe wrapper for enum values - eliminates object usage."""

    enum_class_name: str
    enum_member_name: str
    enum_member_value: str

    @classmethod
    def from_enum(cls, enum_instance: Enum) -> "EnumValue":
        """Create from enum instance."""
        return cls(
            enum_class_name=enum_instance.__class__.__name__,
            enum_member_name=enum_instance.name,
            enum_member_value=str(enum_instance.value),
        )

    @classmethod
    def from_string_representation(cls, value: str) -> "EnumValue":
        """Create from string representation of unknown enum."""
        return cls(
            enum_class_name="UnknownEnum",
            enum_member_name="unknown",
            enum_member_value=value,
        )

    def matches_enum_type(self, enum_class: type[Enum]) -> bool:
        """Check if this value matches the given enum class."""
        return self.enum_class_name == enum_class.__name__

    def to_string(self) -> str:
        """Convert to string representation."""
        return (
            f"{self.enum_class_name}.{self.enum_member_name}={self.enum_member_value}"
        )
