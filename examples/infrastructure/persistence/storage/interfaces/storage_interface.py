from abc import ABC, abstractmethod

from infrastructure.persistence.storage.entities.storage_entry import StorageEntry


class StorageInterface(ABC):
    """Interface for storage services."""

    @abstractmethod
    def save_file(self, path: str, data: bytes) -> None:
        """Saves a file to storage."""
        pass

    @abstractmethod
    def write_text_file(self, path: str, content: str, encoding: str) -> None:
        """Writes a text file to storage."""
        pass

    @abstractmethod
    def read_file(self, path: str) -> bytes:
        """Reads a file from storage."""
        pass

    @abstractmethod
    def read_text_file(self, path: str, encoding: str) -> str:
        """Reads a text file from storage."""
        pass

    @abstractmethod
    def delete_file(self, path: str) -> None:
        """Deletes a file from storage."""
        pass

    @abstractmethod
    def file_exists(self, path: str) -> bool:
        """Checks if a file exists in storage."""
        pass

    @abstractmethod
    def list_directory(self, path: str) -> list[StorageEntry]:
        """Lists a directory in storage."""
        pass

    @abstractmethod
    def make_directory(self, path: str) -> None:
        """Creates a directory in storage."""
        pass

    @abstractmethod
    def remove_directory(self, path: str) -> None:
        """Removes a directory from storage."""
        pass

    @abstractmethod
    def move_file(self, src_path: str, dest_path: str) -> None:
        """Move a file from source to destination path."""
        pass

    @abstractmethod
    def copy_file(self, src_path: str, dest_path: str) -> None:
        """Copy a file from source to destination path."""
        pass

    @abstractmethod
    def get_file_metadata(self, path: str) -> StorageEntry:
        """Get metadata information for a file at the given path."""
        pass
