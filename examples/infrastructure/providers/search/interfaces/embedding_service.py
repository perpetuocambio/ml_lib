"""Domain interface for embedding services."""

from abc import ABC, abstractmethod
from collections.abc import Sequence


class IEmbeddingService(ABC):
    """Domain interface for text embedding services."""

    @abstractmethod
    async def embed_text(self, text: str) -> Sequence[float]:
        """Generate embeddings for text."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced."""
        pass
