"""Collection of query results for database operations."""

from __future__ import annotations

from dataclasses import dataclass

from infrastructure.persistence.database.entities.query_result import QueryResult


@dataclass(frozen=True)
class QueryResultsCollection:
    """Type-safe collection of query results."""

    items: list[QueryResult]

    def __post_init__(self) -> None:
        """Validate collection contents."""
        if not all(isinstance(item, QueryResult) for item in self.items):
            raise ValueError("All items must be QueryResult instances")

    @classmethod
    def create(cls, results: list[dict]) -> QueryResultsCollection:
        """Create collection from list of dicts."""
        query_results = [QueryResult.create(result) for result in results]
        return cls(items=query_results)

    @classmethod
    def empty(cls) -> QueryResultsCollection:
        """Create empty collection."""
        return cls(items=())

    def is_empty(self) -> bool:
        """Check if collection is empty."""
        return len(self.items) == 0

    def count(self) -> int:
        """Get count of items."""
        return len(self.items)

    def first(self) -> QueryResult | None:
        """Get first result or None if empty."""
        return self.items[0] if self.items else None
