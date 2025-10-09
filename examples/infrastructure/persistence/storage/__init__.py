"""Storage infrastructure module."""

from infrastructure.persistence.storage.interfaces.storage_interface import (
    StorageInterface,
)
from infrastructure.persistence.storage.services.local_storage_service import (
    LocalStorageService,
)

__all__ = [
    "StorageInterface",
    "LocalStorageService",
]
