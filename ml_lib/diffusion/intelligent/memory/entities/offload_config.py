"""Model offloading configuration entities."""

from dataclasses import dataclass
from enum import Enum


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
