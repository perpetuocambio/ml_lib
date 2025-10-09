"""Search availability status - replaces dict with typed classes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchAvailabilityStatus:
    """Search availability status for different providers - replaces dict with typed classes."""

    web_search: bool = False
    document_search: bool = False
    semantic_search: bool = False

    @property
    def any_available(self) -> bool:
        """Check if any search provider is available."""
        return self.web_search or self.document_search or self.semantic_search

    @property
    def total_providers(self) -> int:
        """Get total number of available providers."""
        return sum([self.web_search, self.document_search, self.semantic_search])

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names."""
        available = []
        if self.web_search:
            available.append("web_search")
        if self.document_search:
            available.append("document_search")
        if self.semantic_search:
            available.append("semantic_search")
        return available
