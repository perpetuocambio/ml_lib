"""System resource detection entities."""

from dataclasses import dataclass


@dataclass
class SystemResources:
    """Detected system resources."""

    total_vram_gb: float
    available_vram_gb: float
    total_ram_gb: float
    available_ram_gb: float

    has_cuda: bool
    has_mps: bool  # Apple Silicon
    cuda_device_count: int = 0

    compute_capability: tuple[int, int] | None = None  # For CUDA

    @property
    def has_gpu(self) -> bool:
        """Check if any GPU is available."""
        return self.has_cuda or self.has_mps

    @property
    def gpu_type(self) -> str:
        """Get GPU type."""
        if self.has_cuda:
            return "CUDA"
        elif self.has_mps:
            return "MPS"
        else:
            return "CPU"

    @property
    def vram_category(self) -> str:
        """Categorize VRAM size."""
        if self.available_vram_gb < 6:
            return "low"  # <6GB
        elif self.available_vram_gb < 12:
            return "medium"  # 6-12GB
        elif self.available_vram_gb < 20:
            return "high"  # 12-20GB
        else:
            return "very_high"  # >20GB
