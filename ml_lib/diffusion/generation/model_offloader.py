"""Model offloading strategies for memory optimization."""

import logging
from typing import Protocol

from ml_lib.system.services.memory_manager import MemoryManager
from ml_lib.diffusion.models import (
    OffloadConfig,
    OffloadStrategy,
)

logger = logging.getLogger(__name__)


class ModelProtocol(Protocol):
    """Protocol for model objects that can be moved between devices."""

    def to(self, **kwargs) -> "ModelProtocol": ...  # type: ignore[misc]


class ModelOffloader:
    """Manages model component offloading between devices."""

    def __init__(
        self,
        strategy: OffloadStrategy = OffloadStrategy.AUTO,
        max_vram_gb: float = 16.0,
        memory_manager: MemoryManager | None = None,
    ):
        """
        Initialize model offloader.

        Args:
            strategy: Offload strategy
            max_vram_gb: Maximum VRAM to use
            memory_manager: Memory manager instance
        """
        self.strategy = strategy
        self.max_vram_gb = max_vram_gb
        self.memory_manager = memory_manager or MemoryManager()

        logger.info(
            f"ModelOffloader initialized: strategy={strategy.value}, "
            f"max_vram={max_vram_gb}GB"
        )

    def get_offload_config(self) -> OffloadConfig:
        """
        Get offload configuration based on strategy.

        Returns:
            OffloadConfig with device assignments
        """
        if self.strategy == OffloadStrategy.AUTO:
            return self._auto_configure()
        elif self.strategy == OffloadStrategy.SEQUENTIAL:
            return self._sequential_configure()
        elif self.strategy == OffloadStrategy.CPU_OFFLOAD:
            return self._cpu_offload_configure()
        elif self.strategy == OffloadStrategy.FULL_GPU:
            return self._full_gpu_configure()
        elif self.strategy == OffloadStrategy.BALANCED:
            return self._balanced_configure()
        else:
            return self._auto_configure()

    def _auto_configure(self) -> OffloadConfig:
        """
        Automatic configuration based on available VRAM.

        Heuristics:
        - <6GB: Sequential loading
        - 6-8GB: UNet GPU, others CPU
        - 8-12GB: UNet + Text Encoder GPU, VAE CPU
        - >12GB: All GPU
        """
        vram = self.memory_manager.resources.available_vram_gb

        logger.info(f"Auto-configuring for {vram:.2f}GB VRAM")

        if vram < 6:
            logger.info("Low VRAM detected, using sequential loading")
            return self._sequential_configure()

        elif vram < 8:
            logger.info("Medium-low VRAM, UNet on GPU, rest on CPU")
            return OffloadConfig(
                unet_device="cuda",
                text_encoder_device="cpu",
                vae_device="cpu",
                lora_device="cuda",
                enable_cpu_offload=True,
            )

        elif vram < 12:
            logger.info("Medium VRAM, UNet + Text Encoder on GPU")
            return OffloadConfig(
                unet_device="cuda",
                text_encoder_device="cuda",
                vae_device="cpu",
                lora_device="cuda",
            )

        else:
            logger.info("High VRAM, all components on GPU")
            return self._full_gpu_configure()

    def _sequential_configure(self) -> OffloadConfig:
        """Sequential loading configuration."""
        return OffloadConfig(
            unet_device="cuda",
            text_encoder_device="cpu",
            vae_device="cpu",
            lora_device="cuda",
            enable_sequential=True,
            clear_after_use=True,
        )

    def _cpu_offload_configure(self) -> OffloadConfig:
        """CPU offload configuration."""
        return OffloadConfig(
            unet_device="cuda",
            text_encoder_device="cpu",
            vae_device="cpu",
            lora_device="cuda",
            enable_cpu_offload=True,
        )

    def _full_gpu_configure(self) -> OffloadConfig:
        """Full GPU configuration."""
        return OffloadConfig(
            unet_device="cuda",
            text_encoder_device="cuda",
            vae_device="cuda",
            lora_device="cuda",
        )

    def _balanced_configure(self) -> OffloadConfig:
        """Balanced configuration."""
        vram = self.memory_manager.resources.available_vram_gb

        # Similar to auto but more conservative
        if vram < 10:
            return OffloadConfig(
                unet_device="cuda",
                text_encoder_device="cpu",
                vae_device="cpu",
                lora_device="cuda",
                enable_cpu_offload=True,
            )
        else:
            return OffloadConfig(
                unet_device="cuda",
                text_encoder_device="cuda",
                vae_device="cpu",  # Always offload VAE to save VRAM
                lora_device="cuda",
            )

    def move_to_device(
        self,
        model: ModelProtocol,
        device: str,
        dtype: str | None = None,
    ) -> ModelProtocol:
        """
        Move model to specified device.

        Args:
            model: Model to move
            device: Target device ("cuda", "cpu", "mps")
            dtype: Optional dtype (torch.float16, torch.float32, etc.)

        Returns:
            Model on target device
        """
        try:
            import torch

            # Handle device string
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"

            # Move model
            if dtype is not None:
                model = model.to(device=device, dtype=dtype)
            else:
                model = model.to(device)

            # Clear cache
            if device == "cpu" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.debug(f"Moved model to {device}" + (f" ({dtype})" if dtype else ""))

            return model

        except Exception as e:
            logger.error(f"Failed to move model to {device}: {e}")
            raise

    def enable_sequential_cpu_offload(
        self, pipeline: PipelineProtocol
    ) -> PipelineProtocol:
        """
        Enable sequential CPU offload for diffusers pipeline.

        Args:
            pipeline: Diffusers pipeline

        Returns:
            Pipeline with offload enabled
        """
        try:
            if hasattr(pipeline, "enable_sequential_cpu_offload"):
                pipeline.enable_sequential_cpu_offload()
                logger.info("Sequential CPU offload enabled")
            else:
                logger.warning("Pipeline does not support sequential CPU offload")

            return pipeline

        except Exception as e:
            logger.error(f"Failed to enable sequential offload: {e}")
            return pipeline

    def enable_model_cpu_offload(self, pipeline: PipelineProtocol) -> PipelineProtocol:
        """
        Enable model-wise CPU offload for diffusers pipeline.

        Args:
            pipeline: Diffusers pipeline

        Returns:
            Pipeline with offload enabled
        """
        try:
            if hasattr(pipeline, "enable_model_cpu_offload"):
                pipeline.enable_model_cpu_offload()
                logger.info("Model CPU offload enabled")
            else:
                logger.warning("Pipeline does not support model CPU offload")

            return pipeline

        except Exception as e:
            logger.error(f"Failed to enable model offload: {e}")
            return pipeline

    def apply_config(
        self, pipeline: PipelineProtocol, config: OffloadConfig
    ) -> PipelineProtocol:
        """
        Apply offload configuration to pipeline.

        Args:
            pipeline: Diffusers pipeline
            config: Offload configuration

        Returns:
            Configured pipeline
        """
        try:
            # Sequential offload takes precedence
            if config.enable_sequential:
                return self.enable_sequential_cpu_offload(pipeline)

            # Model-wise CPU offload
            if config.enable_cpu_offload:
                return self.enable_model_cpu_offload(pipeline)

            # Manual device assignment
            if hasattr(pipeline, "unet") and pipeline.unet is not None:
                pipeline.unet = self.move_to_device(pipeline.unet, config.unet_device)

            if hasattr(pipeline, "text_encoder") and pipeline.text_encoder is not None:
                pipeline.text_encoder = self.move_to_device(
                    pipeline.text_encoder, config.text_encoder_device
                )

            if hasattr(pipeline, "vae") and pipeline.vae is not None:
                pipeline.vae = self.move_to_device(pipeline.vae, config.vae_device)

            logger.info("Offload configuration applied to pipeline")

            return pipeline

        except Exception as e:
            logger.error(f"Failed to apply offload config: {e}")
            return pipeline
