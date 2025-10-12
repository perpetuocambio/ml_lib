"""Protocols for memory management."""

from typing import Protocol, runtime_checkable

from PIL.Image import Image as PILImage

from ml_lib.system.resource_monitor import SystemResources


@runtime_checkable
class PipelineOutputProtocol(Protocol):
    """Protocol for pipeline output objects."""

    @property
    def images(self) -> list[PILImage]:
        """List of generated PIL Images."""
        ...


@runtime_checkable
class DiffusionPipelineProtocol(Protocol):
    """Protocol for diffusion pipeline objects (from diffusers library)."""

    def to(self, device: str) -> "DiffusionPipelineProtocol":
        """Move pipeline to device."""
        ...

    def __call__(self, **kwargs) -> PipelineOutputProtocol:
        """Generate images."""
        ...


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
    def resources(self) -> SystemResources:
        """Get system resources information."""
        ...


@runtime_checkable
class ModelOffloaderProtocol(Protocol):
    """Protocol for model offloader implementations."""

    def offload_model(self, model_id: str) -> None:
        """
        Offload a model from VRAM.

        Args:
            model_id: Model to offload
        """
        ...

    def load_model(self, model_id: str) -> None:
        """
        Load a model to VRAM.

        Args:
            model_id: Model to load
        """
        ...


@runtime_checkable
class MemoryOptimizerProtocol(Protocol):
    """Protocol for memory optimizer implementations."""

    def optimize_pipeline(self, pipeline: DiffusionPipelineProtocol) -> None:
        """
        Apply memory optimizations to a pipeline.

        Args:
            pipeline: Diffusion pipeline to optimize
        """
        ...

    def cleanup_after_model_load(self) -> None:
        """Cleanup memory after loading a model."""
        ...

    def cleanup_after_generation(self) -> None:
        """Cleanup memory after image generation."""
        ...
