"""Resource monitoring interface - Ports & Adapters pattern.

This interface defines the contract for resource monitoring.
Diffusion domain depends on this abstraction, not concrete implementation.
"""

from typing import Protocol, runtime_checkable
from dataclasses import dataclass


@dataclass
class ResourceStats:
    """System resource statistics snapshot."""

    timestamp: float
    available_vram_gb: float
    total_vram_gb: float
    used_vram_gb: float
    available_ram_gb: float
    total_ram_gb: float
    cpu_usage_percent: float
    has_gpu: bool

    @property
    def vram_usage_percent(self) -> float:
        """Calculate VRAM usage percentage."""
        if self.total_vram_gb == 0:
            return 0.0
        return (self.used_vram_gb / self.total_vram_gb) * 100

    def can_fit_model(self, estimated_size_gb: float, buffer_percent: float = 0.1) -> bool:
        """
        Check if model can fit in available VRAM.

        Args:
            estimated_size_gb: Estimated model size
            buffer_percent: Safety buffer (default 10%)

        Returns:
            True if model should fit
        """
        required_size = estimated_size_gb * (1 + buffer_percent)
        return self.available_vram_gb >= required_size


@runtime_checkable
class IResourceMonitor(Protocol):
    """
    Protocol for system resource monitoring.

    Implementations must provide current resource stats.
    """

    def get_current_stats(self) -> ResourceStats:
        """
        Get current system resource snapshot.

        Returns:
            ResourceStats with current state
        """
        ...

    def can_fit_model(self, estimated_size_gb: float, device: str = "cuda") -> bool:
        """
        Check if a model of given size can fit.

        Args:
            estimated_size_gb: Estimated model size in GB
            device: Target device ("cuda" or "cpu")

        Returns:
            True if model should fit
        """
        ...
