"""Collection of search results for domain operations."""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.providers.search.entities.search_result import SearchResult


@dataclass(frozen=True)
class SearchResultsCollection:
    """Type-safe collection of search results."""

    items: list[SearchResult]

    def __post_init__(self) -> None:
        """Validate collection contents."""
        if not all(isinstance(item, SearchResult) for item in self.items):
            raise ValueError("All items must be SearchResult instances")

    @classmethod
    def create(cls, results: list[SearchResult]) -> SearchResultsCollection:
        """Create collection from list of results."""
        return cls(items=list(results))

    @classmethod
    def empty(cls) -> SearchResultsCollection:
        """Create empty collection."""
        return cls(items=[])

    def is_empty(self) -> bool:
        """Check if collection is empty."""
        return len(self.items) == 0

    def count(self) -> int:
        """Get count of items."""
        return len(self.items)
