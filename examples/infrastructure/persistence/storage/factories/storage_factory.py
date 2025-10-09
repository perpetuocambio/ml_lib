"""Storage factory implementation."""

from infrastructure.persistence.storage.interfaces.storage_factory_interface import (
    IStorageFactory,
)
from infrastructure.persistence.storage.interfaces.storage_interface import (
    StorageInterface,
)
from infrastructure.persistence.storage.services.cloud_storage_service import (
    CloudStorageService,
)
from infrastructure.persistence.storage.services.local_storage_service import (
    LocalStorageService,
)


class StorageFactory(IStorageFactory):
    """Factory for creating storage implementations."""

    def create_local_storage(self) -> StorageInterface:
        """Create local storage implementation."""
        return LocalStorageService()

    def create_cloud_storage(self, **config) -> StorageInterface:
        """Create cloud storage implementation."""
        return CloudStorageService(**config)

    def create_default_storage(self) -> StorageInterface:
        """Create default storage implementation (local)."""
        return self.create_local_storage()
