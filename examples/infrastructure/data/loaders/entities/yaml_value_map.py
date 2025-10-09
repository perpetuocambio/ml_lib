"""YAML value mapping for data loading."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class YamlValueMap:
    """Type-safe YAML value mapping."""

    string_values: list[tuple[str, str]] = field(default_factory=list)
    numeric_values: list[tuple[str, float]] = field(default_factory=list)
    boolean_values: list[tuple[str, bool]] = field(default_factory=list)
    nested_maps: list[tuple[str, YamlValueMap]] = field(default_factory=list)

    def add_string(self, key: str, value: str) -> None:
        """Add string value."""
        self.string_values.append((key, value))

    def add_numeric(self, key: str, value: float) -> None:
        """Add numeric value."""
        self.numeric_values.append((key, value))

    def add_boolean(self, key: str, value: bool) -> None:
        """Add boolean value."""
        self.boolean_values.append((key, value))

    def add_nested_map(self, key: str, value: YamlValueMap) -> None:
        """Add nested mapping."""
        self.nested_maps.append((key, value))

    def get_string(self, key: str) -> str | None:
        """Get string value by key."""
        for k, v in self.string_values:
            if k == key:
                return v
        return None

    def get_numeric(self, key: str) -> float | None:
        """Get numeric value by key."""
        for k, v in self.numeric_values:
            if k == key:
                return v
        return None

    def get_boolean(self, key: str) -> bool | None:
        """Get boolean value by key."""
        for k, v in self.boolean_values:
            if k == key:
                return v
        return None
