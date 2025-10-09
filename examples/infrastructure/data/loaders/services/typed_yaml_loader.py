"""
Simple TypedYamlLoader - YAML to dataclass casting without over-engineering.

Uses native Python types instead of custom wrapper classes.
Follows CLAUDE.md principles: no unnecessary abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TypeVar, Union, get_args, get_origin

import yaml
from infrastructure.data.loaders.entities.yaml_raw_data import YamlRawData, YamlValue
from infrastructure.serialization.protocol_serializer import ProtocolSerializer

T = TypeVar("T")


@dataclass
class TypedYamlLoader:
    """Simple YAML loader that casts directly to dataclass types."""

    file_path: str
    target_type: type[T]

    def load(self) -> T:
        """Load YAML file and cast to target dataclass type."""
        path = Path(self.file_path)

        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")

        if not is_dataclass(self.target_type):
            raise TypeError(f"{self.target_type} must be a dataclass")

        with open(path, encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f)

        # Wrap in typed container - eliminates dict parameter exposure
        raw_data = YamlRawData.from_yaml_content(yaml_content)
        return self._cast_to_dataclass(raw_data, self.target_type)

    def _cast_to_dataclass(self, data: YamlRawData, target_type: type[T]) -> T:
        """Cast YAML data to target dataclass using typed wrapper."""
        if not data.is_mapping_like():
            raise ValueError(
                f"Expected mapping for {target_type.__name__}, got scalar or sequence"
            )

        # Use ProtocolSerializer for infrastructure boundary operations
        field_data = ProtocolSerializer.serialize_dict_data()
        dataclass_fields = fields(target_type)

        for field in dataclass_fields:
            field_name = field.name
            raw_value_wrapper = data.get_field_value(field_name)
            raw_value = (
                raw_value_wrapper.get_scalar_value() if raw_value_wrapper else None
            )

            if raw_value is not None:
                try:
                    casted_value = self._cast_value_to_type(raw_value, field.type)
                    field_data[field_name] = casted_value
                except Exception as e:
                    raise ValueError(
                        f"Cannot cast field '{field_name}' in {target_type.__name__}: {e}"
                    ) from e
            elif field.default != field.default_factory:
                field_data[field_name] = field.default
            else:
                raise ValueError(
                    f"Required field '{field_name}' missing in {target_type.__name__}"
                )

        return target_type(**field_data)

    def _cast_value_to_type(self, value: YamlValue, target_type: type) -> YamlValue:
        """Cast value to target type using native Python types."""
        if value is None:
            return None

        # Handle complex types
        origin = get_origin(target_type)
        args = get_args(target_type)

        # Handle List[X]
        if origin is list and args:
            if not isinstance(value, list):
                raise ValueError(f"Expected list for {target_type}, got {type(value)}")
            element_type = args[0]
            return [self._cast_value_to_type(item, element_type) for item in value]

        # Handle Union (includes Optional[X])
        if origin is Union:
            if value is None and type(None) in args:
                return None
            # Try each union type
            for union_type in args:
                if union_type is type(None):
                    continue
                try:
                    return self._cast_value_to_type(value, union_type)
                except Exception:
                    continue
            raise ValueError(f"Cannot cast {type(value)} to {target_type}")

        # Handle nested dataclass
        if is_dataclass(target_type):
            nested_data = YamlRawData.from_yaml_content(value)
            return self._cast_to_dataclass(nested_data, target_type)

        # Handle primitives
        if target_type is str:
            return str(value)
        elif target_type is int:
            return int(value)
        elif target_type is float:
            return float(value)
        elif target_type is bool:
            return bool(value)

        # Default: return as-is
        return value
