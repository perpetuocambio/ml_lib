"""Type-safe YAML data container with generic conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from infrastructure.serialization.protocol_serializer import ProtocolSerializer
from infrastructure.data.loaders.entities.yaml_convertible import YamlStructure

T = TypeVar("T")


@dataclass(frozen=True)
class TypedYamlData(Generic[T]):
    """Type-safe YAML data container with generic conversion."""

    data: YamlStructure

    @classmethod
    def load_from_content(
        cls, yaml_content: YamlStructure
    ) -> TypedYamlData[YamlStructure]:
        """Load from raw YAML content with type safety."""
        return cls(data=yaml_content)

    def convert_to(self, target_type: type[T]) -> T:
        """Convert YAML data to specific typed object."""
        if not isinstance(self.data, dict):
            raise ValueError(f"Cannot convert non-dict YAML data to {target_type}")

        # If target type implements YamlConvertible protocol
        if hasattr(target_type, "from_yaml_dict"):
            return target_type.from_yaml_dict(self.data)

        # If target type is a dataclass, use field mapping
        if hasattr(target_type, "__dataclass_fields__"):
            return self._convert_to_dataclass(target_type)

        raise ValueError(f"Cannot convert to unsupported type: {target_type}")

    def _convert_to_dataclass(self, target_type: type[T]) -> T:
        """Convert to dataclass using field mapping."""
        if not isinstance(self.data, dict):
            raise ValueError("Cannot convert non-dict data to dataclass")

        # Use ProtocolSerializer for infrastructure boundary operations
        kwargs = ProtocolSerializer.serialize_dict_data()
        data_dict = ProtocolSerializer.serialize_dict_data(self.data)
        for field_name, field_info in target_type.__dataclass_fields__.items():
            if field_name in data_dict:
                kwargs[field_name] = self._convert_field_value(
                    data_dict[field_name], field_info.type
                )

        return target_type(**kwargs)

    def _convert_field_value(self, value: YamlStructure, field_type: type) -> any:
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

    def is_dict(self) -> bool:
        """Check if data is dictionary-like."""
        return isinstance(self.data, dict)

    def is_list(self) -> bool:
        """Check if data is list-like."""
        return isinstance(self.data, list)

    def is_primitive(self) -> bool:
        """Check if data is a primitive value."""
        return isinstance(self.data, str | int | float | bool | type(None))

    def get_keys(self) -> list[str]:
        """Get keys if data is dictionary."""
        if isinstance(self.data, dict):
            # Use ProtocolSerializer for infrastructure boundary operations
            data_dict = ProtocolSerializer.serialize_dict_data(self.data)
            return list(data_dict.keys())
        return []

    def get_field(self, key: str) -> TypedYamlData[YamlStructure]:
        """Get field from dictionary data."""
        if isinstance(self.data, dict):
            # Use ProtocolSerializer for infrastructure boundary operations
            data_dict = ProtocolSerializer.serialize_dict_data(self.data)
            if key in data_dict:
                return TypedYamlData.load_from_content(data_dict[key])
        raise KeyError(f"Field '{key}' not found in YAML data")


# Factory function for type-safe YAML loading
def load_yaml_as(yaml_content: YamlStructure, target_type: type[T]) -> T:
    """Load YAML content directly as specific type."""
    typed_data = TypedYamlData.load_from_content(yaml_content)
    return typed_data.convert_to(target_type)
