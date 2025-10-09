from infrastructure.persistence.storage.interfaces.storage_interface import (
    StorageInterface,
)


class StorageCopyHandler:
    """Handler for copying files between storage services."""

    def __init__(self, source: StorageInterface, destination: StorageInterface) -> None:
        """Initializes the handler with source and destination storage services."""
        self.source = source
        self.destination = destination

    def copy_file(self, path: str) -> None:
        """Copies a file from the source to the destination storage service."""
        data = self.source.read_file(path)
        self.destination.save_file(path, data)
