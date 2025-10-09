"""HTTP header name-value pair for library compatibility."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HttpHeaderPair:
    """HTTP header name-value pair for library compatibility."""

    name: str
    value: str

    def get_name(self) -> str:
        """Get header name."""
        return self.name

    def get_value(self) -> str:
        """Get header value."""
        return self.value
