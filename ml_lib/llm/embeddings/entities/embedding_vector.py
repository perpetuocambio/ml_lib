"""Embedding vector value object."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingVector:
    """Type-safe embedding vector."""

    values: list[float]

    def __post_init__(self) -> None:
        """Validate embedding vector."""
        if not self.values:
            raise ValueError("Embedding vector cannot be empty")
        if not all(isinstance(val, int | float) for val in self.values):
            raise ValueError("All values must be numeric")

    @classmethod
    def create(cls, values: list[float]) -> EmbeddingVector:
        """Create embedding from list of floats."""
        return cls(values=list(values))

    def dimension(self) -> int:
        """Get embedding dimension."""
        return len(self.values)

    def to_list(self) -> list[float]:
        """Convert to list for compatibility."""
        return list(self.values)
