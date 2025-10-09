from infrastructure.persistence.storage.entities.storage_entry import StorageEntry
from infrastructure.persistence.storage.interfaces.storage_interface import (
    StorageInterface,
)


class CloudStorageService(StorageInterface):
    """Service for interacting with cloud storage."""

    def save_file(self, path: str, data: bytes) -> None:
        """Saves a file to cloud storage."""
        raise NotImplementedError("Cloud storage not implemented yet.")

    def write_text_file(self, path: str, content: str, encoding: str) -> None:
        """Writes a text file to cloud storage."""
        raise NotImplementedError("Cloud storage not implemented yet.")

    def read_file(self, path: str) -> bytes:
        """Reads a file from cloud storage."""
        raise NotImplementedError("Cloud storage not implemented yet.")

    def read_text_file(self, path: str, encoding: str) -> str:
        """Reads a text file from cloud storage."""
        raise NotImplementedError("Cloud storage not implemented yet.")

    def delete_file(self, path: str) -> None:
        """Deletes a file from cloud storage."""
        raise NotImplementedError("Cloud storage not implemented yet.")

    def file_exists(self, path: str) -> bool:
        """Checks if a file exists in cloud storage."""
        raise NotImplementedError("Cloud storage not implemented yet.")

    def list_directory(self, path: str) -> list[StorageEntry]:
        """Lists a directory in cloud storage."""
        raise NotImplementedError("Cloud storage not implemented yet.")

    def make_directory(self, path: str) -> None:
        """Creates a directory in cloud storage."""
        raise NotImplementedError("Cloud storage not implemented yet.")

    def remove_directory(self, path: str) -> None:
        """Removes a directory from cloud storage."""
        raise NotImplementedError("Cloud storage not implemented yet.")

    def move_file(self, src_path: str, dest_path: str) -> None:
        """Moves a file in cloud storage."""
        raise NotImplementedError("Cloud storage not implemented yet.")

    def copy_file(self, src_path: str, dest_path: str) -> None:
        """Copies a file in cloud storage."""
        raise NotImplementedError("Cloud storage not implemented yet.")

    def get_file_metadata(self, path: str) -> StorageEntry:
        """Gets file metadata from cloud storage."""
        raise NotImplementedError("Cloud storage not implemented yet.")
