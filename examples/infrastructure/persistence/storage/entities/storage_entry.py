from dataclasses import dataclass
from datetime import datetime


@dataclass
class StorageEntry:
    """Represents a file or directory entry in storage."""

    name: str
    is_dir: bool
    size: int
    modified: datetime

    def is_file(self) -> bool:
        """Check if entry is a file."""
        return not self.is_dir

    def is_directory(self) -> bool:
        """Check if entry is a directory."""
        return self.is_dir

    def get_size_bytes(self) -> int:
        """Get size in bytes."""
        return self.size

    def get_name(self) -> str:
        """Get entry name."""
        return self.name

    def get_modified_time(self) -> datetime:
        """Get modification time."""
        return self.modified
