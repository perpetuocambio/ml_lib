"""HttpHeaders - Typed representation of HTTP headers."""

from __future__ import annotations

from dataclasses import dataclass, field

from infrastructure.communication.http.entities.http_header_entry import HttpHeaderEntry
from infrastructure.communication.http.interfaces.headers_like_object import (
    HeadersLikeObject,
)


@dataclass
class HttpHeaders:
    """Typed representation of HTTP headers.

    Uses typed list structure instead of dict to comply with project constraints.
    """

    entries: list[HttpHeaderEntry] = field(default_factory=list)

    def set(self, name: str, value: str) -> HttpHeaders:
        """Set a header value.

        Args:
            name: Header name
            value: Header value

        Returns:
            self for method chaining
        """
        # Remove existing header with same name (case-insensitive)
        self.entries = [e for e in self.entries if e.name.lower() != name.lower()]
        self.entries.append(HttpHeaderEntry(name=name, value=value))
        return self

    def get(self, name: str, default: str | None = None) -> str | None:
        """Get a header value.

        Args:
            name: Header name
            default: Value to return if header not found

        Returns:
            Header value or default if not found
        """
        for entry in self.entries:
            if entry.name.lower() == name.lower():
                return entry.value
        return default

    def has(self, name: str) -> bool:
        """Check if a header exists.

        Args:
            name: Header name

        Returns:
            True if header exists
        """
        return any(entry.name.lower() == name.lower() for entry in self.entries)

    def remove(self, name: str) -> HttpHeaders:
        """Remove a header if it exists.

        Args:
            name: Header name

        Returns:
            self for method chaining
        """
        self.entries = [e for e in self.entries if e.name.lower() != name.lower()]
        return self

    @staticmethod
    def from_requests_headers(headers_obj: HeadersLikeObject) -> HttpHeaders:
        """Create from requests response headers.

        Args:
            headers_obj: Headers object from requests response

        Returns:
            New HttpHeaders instance
        """
        entries = []
        for name, value in headers_obj.items():
            entries.append(HttpHeaderEntry(name=name, value=value))
        return HttpHeaders(entries=entries)
