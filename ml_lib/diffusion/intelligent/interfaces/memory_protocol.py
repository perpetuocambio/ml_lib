"""Protocols for memory management."""

from typing import Protocol, Optional, runtime_checkable


@runtime_checkable
class MemoryManagerProtocol(Protocol):
    """Protocol for memory manager implementations."""

    def get_peak_vram_usage(self) -> float:
        """
        Get peak VRAM usage in GB.

        Returns:
            Peak VRAM usage
        """
        ...

    @property
    def resources(self):
        """Get system resources information."""
        ...


@runtime_checkable
class ModelOffloaderProtocol(Protocol):
    """Protocol for model offloader implementations."""

    def offload_model(self, model_id: str):
        """
        Offload a model from VRAM.

        Args:
            model_id: Model to offload
        """
        ...

    def load_model(self, model_id: str):
        """
        Load a model to VRAM.

        Args:
            model_id: Model to load
        """
        ...


@runtime_checkable
class MemoryOptimizerProtocol(Protocol):
    """Protocol for memory optimizer implementations."""

    def optimize_pipeline(self, pipeline):
        """
        Apply memory optimizations to a pipeline.

        Args:
            pipeline: Diffusion pipeline to optimize
        """
        ...

    def cleanup_after_model_load(self):
        """Cleanup memory after loading a model."""
        ...

    def cleanup_after_generation(self):
        """Cleanup memory after image generation."""
        ...
