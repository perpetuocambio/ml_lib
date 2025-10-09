"""Type-safe configuration data mappings."""

from dataclasses import dataclass

from infrastructure.config.types.simple_config_data_entry import SimpleConfigDataEntry
from infrastructure.serialization.protocol_serializer import ProtocolSerializer


@dataclass(frozen=True)
class ConfigDataMap:
    """Type-safe container for configuration data - replaces dict[str, str]."""

    entries: list[SimpleConfigDataEntry]

    @classmethod
    def from_protocol_data(
        cls,
        data: str | int | float | bool | list[str],
        protocol_serializer: ProtocolSerializer,
    ) -> "ConfigDataMap":
        """Create from protocol data using ProtocolSerializer - NO direct dict usage."""
        # Use ProtocolSerializer for all dict conversions
        return protocol_serializer.deserialize_config_data(data, cls)

    @classmethod
    def empty(cls) -> "ConfigDataMap":
        """Create empty config data map."""
        return cls(entries=[])

    def get_value(self, key: str) -> str | None:
        """Get value by key."""
        for entry in self.entries:
            if entry.key == key:
                return entry.value
        return None

    def has_key(self, key: str) -> bool:
        """Check if key exists."""
        return any(entry.key == key for entry in self.entries)

    def get_all_keys(self) -> list[str]:
        """Get all keys."""
        return [entry.key for entry in self.entries]

    def get_all_entries(self) -> list[SimpleConfigDataEntry]:
        """Get all entries."""
        return self.entries.copy()

    def to_protocol_data(
        self, protocol_serializer: ProtocolSerializer
    ) -> str | int | float | bool | list[str]:
        """Convert to protocol data using ProtocolSerializer - NO direct dict usage."""
        # Use ProtocolSerializer for all dict conversions
        return protocol_serializer.serialize_config_data(self)
