"""Simple configuration data entry for string mappings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SimpleConfigDataEntry:
    """Single configuration data entry for string values."""

    key: str
    value: str
