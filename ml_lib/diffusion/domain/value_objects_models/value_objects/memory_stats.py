"""Value objects for memory optimization."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MemoryStatistics:
    """Memory statistics replacing dict[str, float]."""

    allocated_gb: float
    reserved_gb: float
    free_gb: float
    total_gb: float = 0.0


class PipelineProtocol(Protocol):
    """Protocol for diffusion pipeline objects."""

    def enable_sequential_cpu_offload(self) -> None: ...

    def enable_model_cpu_offload(self) -> None: ...

    def enable_attention_slicing(self, slice_size: int) -> None: ...

    def enable_xformers_memory_efficient_attention(self) -> None: ...


class VAEProtocol(Protocol):
    """Protocol for VAE component."""

    def enable_tiling(self) -> None: ...

    def enable_slicing(self) -> None: ...

    def enable_layerwise_casting(
        self, storage_dtype: object, compute_dtype: object
    ) -> None: ...


class UNetProtocol(Protocol):
    """Protocol for UNet component."""

    def enable_forward_chunking(self) -> None: ...


class TransformerProtocol(Protocol):
    """Protocol for Transformer component (FLUX, etc.)."""

    def enable_layerwise_casting(
        self, storage_dtype: object, compute_dtype: object
    ) -> None: ...


class ModelComponentProtocol(Protocol):
    """Generic protocol for model components."""

    pass
