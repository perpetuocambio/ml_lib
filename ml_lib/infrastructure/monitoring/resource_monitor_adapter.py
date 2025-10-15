"""Adapter that makes ResourceMonitor implement IResourceMonitor protocol.

This adapter bridges the concrete ResourceMonitor implementation
with the domain interface, following Hexagonal Architecture.
"""

from ml_lib.diffusion.domain.interfaces.resource_monitor import IResourceMonitor, ResourceStats
from ml_lib.infrastructure.monitoring.resource_monitor import ResourceMonitor


class ResourceMonitorAdapter(IResourceMonitor):
    """
    Adapter for ResourceMonitor to implement IResourceMonitor protocol.

    Delegates to the concrete ResourceMonitor implementation
    while adapting its interface to match domain expectations.
    """

    def __init__(self, monitor: ResourceMonitor | None = None):
        """
        Initialize adapter.

        Args:
            monitor: ResourceMonitor instance (None = create new)
        """
        self._monitor = monitor if monitor is not None else ResourceMonitor()

    def get_current_stats(self) -> ResourceStats:
        """
        Get current system resource snapshot.

        Returns:
            ResourceStats (domain model)
        """
        # Get stats from concrete implementation
        sys_resources = self._monitor.get_current_stats()

        # Get primary GPU stats (or defaults if no GPU)
        gpu = sys_resources.get_primary_gpu()
        if gpu:
            available_vram_gb = gpu.memory_free_gb
            total_vram_gb = gpu.memory_total_gb
            used_vram_gb = gpu.memory_used_gb
            has_gpu = True
        else:
            available_vram_gb = 0.0
            total_vram_gb = 0.0
            used_vram_gb = 0.0
            has_gpu = False

        # Convert to domain model
        return ResourceStats(
            timestamp=sys_resources.timestamp,
            available_vram_gb=available_vram_gb,
            total_vram_gb=total_vram_gb,
            used_vram_gb=used_vram_gb,
            available_ram_gb=sys_resources.ram.available_gb,
            total_ram_gb=sys_resources.ram.total_gb,
            cpu_usage_percent=sys_resources.cpu.usage_percent,
            has_gpu=has_gpu,
        )

    def can_fit_model(self, estimated_size_gb: float, device: str = "cuda") -> bool:
        """
        Check if a model of given size can fit.

        Args:
            estimated_size_gb: Estimated model size in GB
            device: Target device ("cuda" or "cpu")

        Returns:
            True if model should fit
        """
        # Delegate directly to concrete implementation
        return self._monitor.can_fit_model(estimated_size_gb, device)
