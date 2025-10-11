"""Memory management models for diffusion system.

This module consolidates all memory-related entities from the intelligent memory subsystem.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


# ==================== System Resources ====================


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


# ==================== Offload Configuration ====================


class OffloadStrategy(Enum):
    """Offload strategy for model components."""
    AUTO = "auto"  # Automatic based on VRAM
    SEQUENTIAL = "sequential"  # Load components on-demand
    CPU_OFFLOAD = "cpu_offload"  # UNet GPU, rest CPU
    FULL_GPU = "full_gpu"  # Everything in GPU
    BALANCED = "balanced"  # Smart distribution


@dataclass
class OffloadConfig:
    """Configuration for model component offloading."""

    unet_device: str = "cuda"
    text_encoder_device: str = "cuda"
    vae_device: str = "cuda"
    lora_device: str = "cuda"

    # Sequential loading settings
    enable_sequential: bool = False
    clear_after_use: bool = True

    # CPU offload settings
    enable_cpu_offload: bool = False
    offload_to_disk: bool = False

    @property
    def all_gpu(self) -> bool:
        """Check if all components are on GPU."""
        return all(
            d == "cuda"
            for d in [
                self.unet_device,
                self.text_encoder_device,
                self.vae_device,
                self.lora_device,
            ]
        )

    @property
    def memory_efficient(self) -> bool:
        """Check if using memory-efficient configuration."""
        return (
            self.enable_sequential
            or self.enable_cpu_offload
            or not self.all_gpu
        )


# ==================== Loaded Model Tracking ====================


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
