"""Search query entity for unified search system."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class SearchQuery:
    """Infrastructure search query specification - uses strings to maintain independence."""

    query_text: str
    search_types: list[str]  # Use strings instead of Domain enums
    project_id: str | None = None
    max_results: int = 10
    language: str = "es"
    region: str = "es-ES"
    time_filter: str | None = None  # "day", "week", "month", "year"
    include_cached: bool = True
    created_at: datetime = None

    def __post_init__(self):
        """Initialize created_at if not provided."""
        if self.created_at is None:
            object.__setattr__(self, "created_at", datetime.now())

    @property
    def query_id(self) -> str:
        """Generate unique query ID."""
        return f"search_{self.created_at.strftime('%Y%m%d_%H%M%S')}_{hash(self.query_text) % 10000}"

    def includes_web_search(self) -> bool:
        """Check if web search is requested."""
        return "web" in self.search_types or "all" in self.search_types

    def includes_document_search(self) -> bool:
        """Check if document search is requested."""
        return "document" in self.search_types or "all" in self.search_types

    def includes_semantic_search(self) -> bool:
        """Check if semantic search is requested."""
        return "semantic" in self.search_types or "all" in self.search_types
