"""Domain interface for document search services."""

from abc import ABC, abstractmethod

from infrastructure.providers.search.entities.search_query import SearchQuery
from infrastructure.providers.search.entities.search_result_collection import (
    SearchResultCollection,
)


class IDocumentSearchService(ABC):
    """Domain interface for document search services."""

    @abstractmethod
    async def search_documents(self, query: SearchQuery) -> SearchResultCollection:
        """Search through documents and return results."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the service is available."""
        pass
