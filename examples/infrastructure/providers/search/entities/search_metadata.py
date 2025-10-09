"""Metadata for search results."""

from dataclasses import dataclass

from infrastructure.providers.search.entities.search_metadata_entry import (
    SearchMetadataEntry,
)


@dataclass(frozen=True)
class SearchMetadata:
    """Typed metadata for search results."""

    entries: list[SearchMetadataEntry]

    @classmethod
    def empty(cls) -> "SearchMetadata":
        """Create empty metadata."""
        return cls(entries=[])

    @classmethod
    def from_pairs(cls, **kwargs: str) -> "SearchMetadata":
        """Create metadata from key-value pairs."""
        entries = [SearchMetadataEntry(key=k, value=v) for k, v in kwargs.items()]
        return cls(entries=entries)

    def get_value(self, key: str) -> str | None:
        """Get value for a specific key."""
        for entry in self.entries:
            if entry.key == key:
                return entry.value
        return None

    def has_key(self, key: str) -> bool:
        """Check if metadata contains a key."""
        return any(entry.key == key for entry in self.entries)
