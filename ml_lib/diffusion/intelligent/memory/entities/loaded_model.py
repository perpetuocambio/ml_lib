"""Loaded model tracking entities."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class EvictionPolicy(Enum):
    """Policy for evicting models from cache."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    SIZE = "size"  # Largest first
    FIFO = "fifo"  # First In First Out


@dataclass
class LoadedModel:
    """Tracking information for a loaded model."""

    model: Any  # The actual model object
    size_gb: float
    loaded_at: datetime
    device: str = "cuda"

    # Usage tracking
    last_accessed: datetime | None = None
    access_count: int = 0

    def update_access(self):
        """Update access tracking."""
        self.last_accessed = datetime.now()
        self.access_count += 1

    @property
    def age_seconds(self) -> float:
        """Time since loaded."""
        return (datetime.now() - self.loaded_at).total_seconds()

    @property
    def idle_seconds(self) -> float:
        """Time since last access."""
        if self.last_accessed:
            return (datetime.now() - self.last_accessed).total_seconds()
        return self.age_seconds
