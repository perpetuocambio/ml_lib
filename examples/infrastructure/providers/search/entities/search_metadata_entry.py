"""Single metadata entry for search results."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchMetadataEntry:
    """Single metadata key-value pair."""

    key: str
    value: str
