"""
Typed HTTP request body - eliminates dict parameter violations.

Follows CLAUDE.md principle: no dict parameters in public APIs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

# Type alias for JSON-serializable values
JsonSerializable = str | int | float | bool | None | list | dict


@dataclass
class HttpRequestBody:
    """Typed representation of HTTP request body."""

    content: str | bytes
    content_type: str = "application/json"

    @classmethod
    def from_json_serializable(
        cls, data: JsonSerializable, content_type: str = "application/json"
    ) -> HttpRequestBody:
        """Create from any JSON-serializable data."""

        if isinstance(data, str | bytes):
            return cls(content=data, content_type=content_type)
        else:
            # Internal dict usage - completely encapsulated
            json_str = json.dumps(data)
            return cls(content=json_str, content_type=content_type)

    @classmethod
    def from_text(cls, text: str, content_type: str = "text/plain") -> HttpRequestBody:
        """Create from text content."""
        return cls(content=text, content_type=content_type)

    @classmethod
    def from_bytes(
        cls, data: bytes, content_type: str = "application/octet-stream"
    ) -> HttpRequestBody:
        """Create from binary data."""
        return cls(content=data, content_type=content_type)

    def get_content_as_string(self) -> str:
        """Get content as string."""
        if isinstance(self.content, bytes):
            return self.content.decode("utf-8")
        return self.content

    def get_content_as_bytes(self) -> bytes:
        """Get content as bytes."""
        if isinstance(self.content, str):
            return self.content.encode("utf-8")
        return self.content
