"""Infrastructure collection for search results."""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.providers.search.entities.search_result import SearchResult


@dataclass(frozen=True)
class SearchResultCollection:
    """Infrastructure collection of search results."""

    results: list[SearchResult]

    def __post_init__(self) -> None:
        """Validate search result collection."""
        if not isinstance(self.results, list):
            raise ValueError("Results must be a list")
        if not all(isinstance(result, SearchResult) for result in self.results):
            raise ValueError("All results must be SearchResult instances")

    @classmethod
    def empty(cls) -> SearchResultCollection:
        """Create empty search result collection."""
        return cls(results=[])

    @classmethod
    def from_results(cls, results: list[SearchResult]) -> SearchResultCollection:
        """Create collection from search results."""
        return cls(results=results)

    def count(self) -> int:
        """Get number of results."""
        return len(self.results)

    def is_empty(self) -> bool:
        """Check if collection is empty."""
        return len(self.results) == 0

    def get_results(self) -> list[SearchResult]:
        """Get all search results."""
        return list(self.results)  # Return copy
