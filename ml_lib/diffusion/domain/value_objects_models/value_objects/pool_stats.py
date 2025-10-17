"""
Pool statistics value objects.

This module provides type-safe pool statistics,
replacing dict returns.
"""

from dataclasses import dataclass
from typing import Protocol


class ModelProtocol(Protocol):
    """Protocol for model objects."""
    pass  # Models can be any type implementing this protocol


@dataclass(frozen=True)
class PoolStatistics:
    """Pool statistics replacing dict return."""

    loaded_count: int
    current_size_gb: float
    max_size_gb: float
    utilization: float  # 0.0 to 1.0
    model_ids: list[str]

    def __post_init__(self) -> None:
        """Validate statistics."""
        if self.loaded_count < 0:
            raise ValueError(f"loaded_count must be >= 0, got {self.loaded_count}")
        if self.current_size_gb < 0:
            raise ValueError(f"current_size_gb must be >= 0, got {self.current_size_gb}")
        if self.max_size_gb < 0:
            raise ValueError(f"max_size_gb must be >= 0, got {self.max_size_gb}")
        if not 0.0 <= self.utilization <= 1.0:
            raise ValueError(f"utilization must be 0.0-1.0, got {self.utilization}")


@dataclass(frozen=True)
class ModelAccessInfo:
    """Access tracking information for a model."""

    model_id: str
    access_time: float
    access_count: int

    def __post_init__(self) -> None:
        """Validate access info."""
        if self.access_count < 0:
            raise ValueError(f"access_count must be >= 0, got {self.access_count}")
