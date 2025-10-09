"""
Single configuration entry with typed key-value pair.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigEntry:
    """Single configuration entry with typed key-value pair."""

    key: str
    value: str | int | float | bool | list[str] | None
