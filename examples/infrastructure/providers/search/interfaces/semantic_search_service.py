"""Domain interface for semantic search services."""

from abc import ABC, abstractmethod

from infrastructure.providers.search.entities.search_query import SearchQuery
from infrastructure.providers.search.entities.search_result_collection import (
    SearchResultCollection,
)


class ISemanticSearchService(ABC):
    """Domain interface for semantic search services."""

    @abstractmethod
    async def semantic_search(self, query: SearchQuery) -> SearchResultCollection:
        """Perform semantic search and return results."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the service is available."""
        pass

    @property
    @abstractmethod
    def embedding_model(self) -> str:
        """Get the embedding model being used."""
        pass
