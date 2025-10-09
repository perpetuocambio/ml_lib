"""Header name-value pair for protocol definitions."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HeaderPair:
    """Single header name-value pair for protocol definition."""

    name: str
    value: str
