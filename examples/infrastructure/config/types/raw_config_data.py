from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from infrastructure.config.types.config_entry import ConfigEntry

if TYPE_CHECKING:
    from infrastructure.serialization.protocol_serializer import ProtocolSerializer


@dataclass(frozen=True)
class RawConfigData:
    """Type-safe container for raw config data - replaces dict."""

    entries: list[ConfigEntry]

    @classmethod
    def from_protocol_data(
        cls,
        data: str | int | float | bool | list[str],
        protocol_serializer: ProtocolSerializer,
    ) -> RawConfigData:
        """Create from protocol data using ProtocolSerializer - NO direct dict usage."""
        # Use ProtocolSerializer for all dict conversions
        return protocol_serializer.deserialize_config_data(data, cls)

    @classmethod
    def empty(cls) -> RawConfigData:
        """Create empty raw config data."""
        return cls(entries=[])

    @classmethod
    def _convert_value(
        cls, value: str | int | float | bool | list[str] | None
    ) -> str | int | float | bool | list[str] | None:
        """Convert unknown value to ConfigValue."""
        # Simplified implementation to avoid circular dependency
        return value

    def get_value(self, key: str) -> str | int | float | bool | list[str] | None:
        """Get value by key."""
        for entry in self.entries:
            if entry.key == key:
                return entry.value
        return None

    def to_protocol_data(
        self, protocol_serializer: ProtocolSerializer
    ) -> str | int | float | bool | list[str]:
        """Convert to protocol data using ProtocolSerializer - NO direct dict usage."""
        # Use ProtocolSerializer for all dict conversions
        return protocol_serializer.serialize_config_data(self)
