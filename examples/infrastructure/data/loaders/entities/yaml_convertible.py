"""Protocol for objects that can be created from YAML data."""

from __future__ import annotations

from typing import Protocol

from infrastructure.data.loaders.entities.yaml_raw_data import YamlRawData


class YamlConvertible(Protocol):
    """Protocol for objects that can be created from YAML data."""

    @classmethod
    def from_yaml_data(cls, data: YamlRawData) -> YamlConvertible:
        """Create instance from type-safe YAML data."""
        ...
