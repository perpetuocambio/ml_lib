"""Single header entry for requests processing."""

from dataclasses import dataclass


@dataclass(frozen=True)
class HeaderEntry:
    """Single header name-value pair."""

    name: str
    value: str
