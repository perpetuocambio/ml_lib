"""Metadata entity for semantic search results."""

from dataclasses import dataclass


@dataclass
class SemanticSearchMetadata:
    """Metadata for semantic search results without using dictionaries."""

    chunk_index: int
    chunk_total: int | None
    embedding_similarity: float
    search_method: str
    document_id: str | None = None
    document_hash: str | None = None
    content_type: str | None = None
    language: str | None = None
    processed_at: str | None = None

    def to_display_text(self) -> str:
        """Convert metadata to human-readable display text."""
        parts = []

        if self.chunk_total:
            parts.append(f"Chunk {self.chunk_index + 1} of {self.chunk_total}")
        else:
            parts.append(f"Chunk {self.chunk_index}")

        parts.append(f"Similarity: {self.embedding_similarity:.2%}")

        if self.search_method:
            parts.append(f"Method: {self.search_method}")

        if self.content_type:
            parts.append(f"Type: {self.content_type}")

        return " | ".join(parts)
