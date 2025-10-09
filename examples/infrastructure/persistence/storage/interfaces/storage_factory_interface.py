"""Interface for storage factory."""

from abc import ABC, abstractmethod

from infrastructure.persistence.storage.interfaces.storage_interface import (
    StorageInterface,
)


class IStorageFactory(ABC):
    """Interface for creating storage implementations."""

    @abstractmethod
    def create_local_storage(self) -> StorageInterface:
        """Create local storage implementation."""
        pass

    @abstractmethod
    def create_cloud_storage(self, **config) -> StorageInterface:
        """Create cloud storage implementation."""
        pass

    @abstractmethod
    def create_default_storage(self) -> StorageInterface:
        """Create default storage implementation."""
        pass
