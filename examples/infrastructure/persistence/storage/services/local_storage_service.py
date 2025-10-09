import os
import shutil
from datetime import datetime

from infrastructure.persistence.storage.entities.storage_entry import StorageEntry
from infrastructure.persistence.storage.interfaces.storage_interface import (
    StorageInterface,
)


class LocalStorageService(StorageInterface):
    """Service for interacting with the local filesystem."""

    def save_file(self, path: str, data: bytes) -> None:
        """Saves a file to the local filesystem."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def write_text_file(self, path: str, content: str, encoding: str) -> None:
        """Writes a text file to the local filesystem."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding=encoding) as f:
            f.write(content)

    def read_file(self, path: str) -> bytes:
        """Reads a file from the local filesystem."""
        with open(path, "rb") as f:
            return f.read()

    def read_text_file(self, path: str, encoding: str) -> str:
        """Reads a text file from the local filesystem."""
        with open(path, encoding=encoding) as f:
            return f.read()

    def delete_file(self, path: str) -> None:
        """Deletes a file from the local filesystem."""
        os.remove(path)

    def file_exists(self, path: str) -> bool:
        """Checks if a file exists in the local filesystem."""
        return os.path.exists(path)

    def list_directory(self, path: str) -> list[StorageEntry]:
        """Lists a directory in the local filesystem."""
        entries = []
        for entry in os.scandir(path):
            stat = entry.stat()
            entries.append(
                StorageEntry(
                    name=entry.name,
                    is_dir=entry.is_dir(),
                    size=stat.st_size,
                    modified=datetime.fromtimestamp(stat.st_mtime),
                )
            )
        return entries

    def make_directory(self, path: str) -> None:
        """Creates a directory in the local filesystem."""
        os.makedirs(path, exist_ok=True)

    def remove_directory(self, path: str) -> None:
        """Removes a directory from the local filesystem."""
        os.rmdir(path)

    def move_file(self, src_path: str, dest_path: str) -> None:
        """Moves a file in the local filesystem."""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        os.rename(src_path, dest_path)

    def copy_file(self, src_path: str, dest_path: str) -> None:
        """Copies a file in the local filesystem."""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src_path, dest_path)

    def get_file_metadata(self, path: str) -> StorageEntry:
        """Gets file metadata from the local filesystem."""
        stat = os.stat(path)
        return StorageEntry(
            name=os.path.basename(path),
            is_dir=os.path.isdir(path),
            size=stat.st_size,
            modified=datetime.fromtimestamp(stat.st_mtime),
        )
