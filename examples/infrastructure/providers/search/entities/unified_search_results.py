"""Unified search results entity aggregating all sources."""

from dataclasses import dataclass
from datetime import datetime

from infrastructure.providers.search.entities.search_query import SearchQuery
from infrastructure.providers.search.entities.search_result import SearchResult


@dataclass(frozen=True)
class UnifiedSearchResults:
    """Aggregated search results from multiple sources."""

    query: SearchQuery
    results: list[SearchResult]
    total_results: int
    search_duration_ms: int
    cached_results_count: int = 0
    errors: list[str] = None
    completed_at: datetime = None

    def __post_init__(self):
        """Initialize completed_at if not provided."""
        if self.completed_at is None:
            object.__setattr__(self, "completed_at", datetime.now())
        if self.errors is None:
            object.__setattr__(self, "errors", [])

    @property
    def web_results(self) -> list[SearchResult]:
        """Get only web search results."""
        return [r for r in self.results if r.search_type == "web"]

    @property
    def document_results(self) -> list[SearchResult]:
        """Get only document search results."""
        return [r for r in self.results if r.search_type == "document"]

    @property
    def semantic_results(self) -> list[SearchResult]:
        """Get only semantic search results."""
        return [r for r in self.results if r.search_type == "semantic"]

    @property
    def top_results(self, limit: int = 5) -> list[SearchResult]:
        """Get top results sorted by relevance."""
        return sorted(self.results, key=lambda r: r.relevance_score, reverse=True)[
            :limit
        ]

    @property
    def has_errors(self) -> bool:
        """Check if search had any errors."""
        return len(self.errors) > 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate of searches."""
        total_searches = len([t for t in self.query.search_types if t != "all"])
        if total_searches == 0:
            return 1.0
        failed_searches = len(self.errors)
        return max(0.0, (total_searches - failed_searches) / total_searches)
