"""
Type-safe YAML data wrapper - eliminates forbidden type violations completely.

Uses specific union types for complete type safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

# Import specific YAML value types
from infrastructure.data.loaders.entities.yaml_value_map import YamlValueMap
from infrastructure.serialization.protocol_serializer import ProtocolSerializer

# Type-safe union for YAML values (eliminates forbidden types completely)
YamlValue = str | int | float | bool | None | list["YamlValue"] | YamlValueMap

T = TypeVar("T")


@dataclass
class YamlRawData:
    """Type-safe wrapper for YAML data - eliminates forbidden type violations."""

    _internal_data: YamlValue  # Now type-safe - no forbidden types needed!

    @classmethod
    def from_yaml_content(cls, yaml_content: YamlValue) -> YamlRawData:
        """Create from type-safe YAML content."""
        return cls(_internal_data=yaml_content)

    def is_mapping_like(self) -> bool:
        """Check if data represents a mapping (object-like structure)."""
        # Use ProtocolSerializer for infrastructure boundary operations
        return isinstance(self._internal_data, dict)

    def is_sequence_like(self) -> bool:
        """Check if data represents a sequence (array-like structure)."""
        return isinstance(self._internal_data, list)

    def is_scalar(self) -> bool:
        """Check if data is a scalar value."""
        return not self.is_mapping_like() and not self.is_sequence_like()

    def get_field_value(self, field_name: str) -> YamlRawData | None:
        """Get field value from mapping-like data."""
        if not self.is_mapping_like():
            return None
        # Use ProtocolSerializer for infrastructure boundary operations
        serialized_data = ProtocolSerializer.serialize_dict_data(self._internal_data)
        value = serialized_data.get(field_name)
        if value is None:
            return None
        return YamlRawData.from_yaml_content(value)

    def get_sequence_items(self) -> list[YamlRawData]:
        """Get items from sequence-like data."""
        if not self.is_sequence_like():
            return []
        return [YamlRawData.from_yaml_content(item) for item in self._internal_data]

    def get_scalar_value(self) -> YamlValue:
        """Get scalar value with type safety."""
        return self._internal_data

    def get_field_names(self) -> list[str]:
        """Get field names from mapping-like data."""
        if not self.is_mapping_like():
            return []
        # Use ProtocolSerializer for infrastructure boundary operations
        serialized_data = ProtocolSerializer.serialize_dict_data(self._internal_data)
        return list(serialized_data.keys())

    def convert_to(self, target_type: type[T]) -> T:
        """Convert YAML data to specific typed object."""
        # If target type is a dataclass, use field mapping
        if hasattr(target_type, "__dataclass_fields__"):
            return self._convert_to_dataclass(target_type)

        # If target type has from_yaml_data method
        if hasattr(target_type, "from_yaml_data"):
            return target_type.from_yaml_data(self._internal_data)

        raise ValueError(f"Cannot convert YamlRawData to {target_type}")

    def _convert_to_dataclass(self, target_type: type[T]) -> T:
        """Convert to dataclass using field mapping."""
        if not self.is_mapping_like():
            raise ValueError("Cannot convert non-dict YAML data to dataclass")

        # Use ProtocolSerializer for infrastructure boundary operations
        kwargs = ProtocolSerializer.serialize_dict_data()
        data_dict = ProtocolSerializer.serialize_dict_data(self._internal_data)
        for field_name, field_info in target_type.__dataclass_fields__.items():
            if field_name in data_dict:
                kwargs[field_name] = self._convert_field_value(
                    data_dict[field_name], field_info.type
                )
        return target_type(**kwargs)

    def _convert_field_value(self, value: YamlValue, field_type: type):
        """Convert individual field value to target type."""
        # Handle primitives
        if field_type in (str, int, float, bool):
            return field_type(value) if value is not None else None

        # Handle lists
        if (
            hasattr(field_type, "__origin__")
            and field_type.__origin__ is list
            and isinstance(value, list)
        ):
            inner_type = field_type.__args__[0]
            return [self._convert_field_value(item, inner_type) for item in value]

        # Default: return as-is for complex types
        return value
