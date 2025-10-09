"""Type-safe HTTP headers container - replaces dict usage."""

from dataclasses import dataclass, field

from infrastructure.config.types.http_header_entry import HttpHeaderEntry
from infrastructure.serialization.protocol_serializer import ProtocolSerializer


@dataclass(frozen=True)
class HttpHeadersContainer:
    """Type-safe container for HTTP headers - replaces dict[str, str]."""

    entries: list[HttpHeaderEntry] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "HttpHeadersContainer":
        """Create empty headers container."""
        return cls(entries=[])

    @classmethod
    def from_protocol_data(
        cls,
        data: str | int | float | bool | list[str],
        protocol_serializer: ProtocolSerializer,
    ) -> "HttpHeadersContainer":
        """Create from protocol data using ProtocolSerializer - NO direct dict usage."""
        # Use ProtocolSerializer for all dict conversions
        return protocol_serializer.deserialize_config_data(data, cls)

    def get_header_value(self, header_name: str) -> str | None:
        """Get header value by name."""
        for entry in self.entries:
            if entry.name.lower() == header_name.lower():
                return entry.value
        return None

    def has_header(self, header_name: str) -> bool:
        """Check if header exists."""
        return any(entry.name.lower() == header_name.lower() for entry in self.entries)

    def get_all_header_names(self) -> list[str]:
        """Get all header names."""
        return [entry.name for entry in self.entries]

    def to_protocol_data(
        self, protocol_serializer: ProtocolSerializer
    ) -> str | int | float | bool | list[str]:
        """Convert to protocol data using ProtocolSerializer - NO direct dict usage."""
        # Use ProtocolSerializer for all dict conversions
        return protocol_serializer.serialize_config_data(self)
