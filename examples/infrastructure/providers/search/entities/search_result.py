"""Search result entity for unified search system."""

from dataclasses import dataclass
from datetime import datetime

from infrastructure.providers.search.entities.search_metadata import SearchMetadata
from infrastructure.providers.search.entities.search_source import SearchSource


@dataclass(frozen=True)
class SearchResult:
    """Individual search result from any source - uses strings to maintain infrastructure independence."""

    title: str
    content: str
    url: str | None
    source: SearchSource
    search_type: str  # Use string instead of Domain enum
    relevance_score: float  # 0.0 to 1.0
    found_at: datetime
    snippet: str | None = None
    metadata: SearchMetadata | None = (
        None  # Additional context (file path, document type, etc.)
    )

    def __post_init__(self):
        """Initialize found_at if not provided."""
        if self.found_at is None:
            object.__setattr__(self, "found_at", datetime.now())

    @property
    def result_id(self) -> str:
        """Generate unique result ID."""
        return f"{self.source.value}_{hash(self.url or self.title) % 10000}"

    @property
    def is_web_result(self) -> bool:
        """Check if result comes from web search."""
        return self.search_type == "web"

    @property
    def is_document_result(self) -> bool:
        """Check if result comes from document search."""
        return self.search_type == "document"

    @property
    def is_semantic_result(self) -> bool:
        """Check if result comes from semantic search."""
        return self.search_type == "semantic"
