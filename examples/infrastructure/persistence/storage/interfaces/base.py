"""
Base interface for storage backends (local, cloud, etc).
"""

from abc import ABC, abstractmethod


class StorageInterface(ABC):
    """Abstract base interface for storage backends (local, cloud, etc)."""

    @abstractmethod
    def save(self, key: str, data: bytes) -> None:
        """Save data to storage using the provided key."""
        pass

    @abstractmethod
    def load(self, key: str) -> bytes:
        """Load data from storage using the provided key."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if data exists in storage for the given key."""
        pass
