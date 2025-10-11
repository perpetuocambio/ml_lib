"""Memory manager for tracking and optimizing resource usage."""

import gc
import logging
from contextlib import contextmanager
from typing import Generator

from ml_lib.diffusion.intelligent.memory.entities import SystemResources

logger = logging.getLogger(__name__)


class MemoryMonitor:
    """Context manager for monitoring memory usage."""

    def __init__(self, memory_manager: "MemoryManager"):
        """Initialize monitor."""
        self.memory_manager = memory_manager
        self.start_vram = 0.0
        self.peak_vram = 0.0
        self.start_ram = 0.0

    def __enter__(self):
        """Start monitoring."""
        self.start_vram = self.memory_manager.get_vram_usage()
        self.start_ram = self.memory_manager.get_ram_usage()
        self.peak_vram = self.start_vram
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring."""
        self.peak_vram = max(self.peak_vram, self.memory_manager.get_peak_vram_usage())


class MemoryManager:
    """Central memory manager for GPU/CPU resource management."""

    def __init__(self):
        """Initialize memory manager."""
        self.resources = self.detect_resources()
        self.monitors: list[MemoryMonitor] = []

        logger.info(
            f"MemoryManager initialized: {self.resources.gpu_type}, "
            f"{self.resources.available_vram_gb:.2f}GB VRAM available"
        )

    def detect_resources(self) -> SystemResources:
        """
        Detect available system resources.

        Returns:
            SystemResources with detected values
        """
        # Try to import torch
        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not available, assuming CPU-only")
            return self._cpu_only_resources()

        # Check CUDA
        if torch.cuda.is_available():
            return self._detect_cuda_resources(torch)

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return self._detect_mps_resources(torch)

        # Fallback to CPU
        return self._cpu_only_resources()

    def _detect_cuda_resources(self, torch) -> SystemResources:
        """Detect CUDA GPU resources."""
        import psutil

        device_props = torch.cuda.get_device_properties(0)
        total_vram = device_props.total_memory / (1024**3)
        allocated_vram = torch.cuda.memory_allocated(0) / (1024**3)
        available_vram = total_vram - allocated_vram

        vm = psutil.virtual_memory()
        total_ram = vm.total / (1024**3)
        available_ram = vm.available / (1024**3)

        compute_cap = torch.cuda.get_device_capability(0)

        return SystemResources(
            total_vram_gb=total_vram,
            available_vram_gb=available_vram,
            total_ram_gb=total_ram,
            available_ram_gb=available_ram,
            has_cuda=True,
            has_mps=False,
            cuda_device_count=torch.cuda.device_count(),
            compute_capability=compute_cap,
        )

    def _detect_mps_resources(self, torch) -> SystemResources:
        """Detect MPS (Apple Silicon) resources."""
        import psutil

        vm = psutil.virtual_memory()
        total_ram = vm.total / (1024**3)
        available_ram = vm.available / (1024**3)

        # MPS shares unified memory
        # Estimate ~70% of available RAM can be used for MPS
        estimated_vram = available_ram * 0.7

        return SystemResources(
            total_vram_gb=estimated_vram,
            available_vram_gb=estimated_vram,
            total_ram_gb=total_ram,
            available_ram_gb=available_ram,
            has_cuda=False,
            has_mps=True,
            cuda_device_count=0,
        )

    def _cpu_only_resources(self) -> SystemResources:
        """Fallback to CPU-only resources."""
        try:
            import psutil

            vm = psutil.virtual_memory()
            total_ram = vm.total / (1024**3)
            available_ram = vm.available / (1024**3)
        except ImportError:
            total_ram = 8.0  # Assume 8GB
            available_ram = 4.0

        return SystemResources(
            total_vram_gb=0.0,
            available_vram_gb=0.0,
            total_ram_gb=total_ram,
            available_ram_gb=available_ram,
            has_cuda=False,
            has_mps=False,
        )

    @contextmanager
    def track_usage(self) -> Generator[MemoryMonitor, None, None]:
        """
        Context manager for tracking memory usage.

        Usage:
            with memory_manager.track_usage() as tracker:
                # do something
                pass
            print(f"Peak VRAM: {tracker.peak_vram}")
        """
        monitor = MemoryMonitor(self)
        self.monitors.append(monitor)
        try:
            yield monitor
        finally:
            pass

    def get_vram_usage(self) -> float:
        """
        Get current VRAM usage in GB.

        Returns:
            Current VRAM usage
        """
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.memory_allocated(0) / (1024**3)
        except ImportError:
            pass

        return 0.0

    def get_peak_vram_usage(self) -> float:
        """
        Get peak VRAM usage in GB.

        Returns:
            Peak VRAM usage since last reset
        """
        try:
            import torch

            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated(0) / (1024**3)
        except ImportError:
            pass

        return 0.0

    def get_ram_usage(self) -> float:
        """
        Get current RAM usage in GB.

        Returns:
            Current RAM usage
        """
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / (1024**3)
        except ImportError:
            return 0.0

    def clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.debug("CUDA cache cleared")
        except ImportError:
            pass

        gc.collect()
        logger.debug("Garbage collection completed")

    def reset_peak_stats(self):
        """Reset peak memory statistics."""
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                logger.debug("Peak memory stats reset")
        except ImportError:
            pass

    def get_available_vram(self) -> float:
        """
        Get currently available VRAM in GB.

        Returns:
            Available VRAM
        """
        total = self.resources.total_vram_gb
        used = self.get_vram_usage()
        return max(total - used, 0.0)

    def can_fit_model(self, estimated_size_gb: float) -> bool:
        """
        Check if a model of given size can fit in VRAM.

        Args:
            estimated_size_gb: Estimated model size in GB

        Returns:
            True if model should fit
        """
        available = self.get_available_vram()
        # Add 10% buffer
        return available >= (estimated_size_gb * 1.1)

    def get_memory_report(self) -> dict:
        """
        Get comprehensive memory report.

        Returns:
            Dictionary with memory statistics
        """
        return {
            "gpu_type": self.resources.gpu_type,
            "total_vram_gb": self.resources.total_vram_gb,
            "available_vram_gb": self.get_available_vram(),
            "used_vram_gb": self.get_vram_usage(),
            "peak_vram_gb": self.get_peak_vram_usage(),
            "total_ram_gb": self.resources.total_ram_gb,
            "used_ram_gb": self.get_ram_usage(),
            "vram_category": self.resources.vram_category,
        }
