"""HTTP header entry for type-safe header management."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HttpHeaderEntry:
    """Single HTTP header entry."""

    name: str
    value: str
