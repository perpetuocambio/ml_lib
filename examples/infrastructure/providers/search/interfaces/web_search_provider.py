"""Domain interface for web search providers."""

from abc import ABC, abstractmethod

from infrastructure.providers.search.entities.search_query import SearchQuery
from infrastructure.providers.search.entities.search_result_collection import (
    SearchResultCollection,
)


class IWebSearchProvider(ABC):
    """Domain interface for web search providers."""

    @abstractmethod
    async def search(self, query: SearchQuery) -> SearchResultCollection:
        """Perform web search and return results."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass
