"""
Memory Optimization Engine - Our Market Value Differentiator

Implements ALL HuggingFace memory optimization techniques:
1. Sequential CPU Offload (leaf-level)
2. Model CPU Offload (component-level)
3. Group Offloading (advanced)
4. VAE Tiling
5. VAE Slicing
6. Attention Slicing
7. xFormers Memory Efficient Attention
8. Forward Chunking
9. Immediate Garbage Collection
10. Torch CUDA Cache Clearing

Philosophy: FREE MEMORY IMMEDIATELY AFTER USE
"""

import gc
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from ml_lib.diffusion.models.value_objects.memory_stats import (
    MemoryStatistics,
    PipelineProtocol,
    VAEProtocol,
    UNetProtocol,
    TransformerProtocol,
    ModelComponentProtocol,
)

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Memory optimization levels."""

    NONE = "none"  # No optimization (fastest, most memory)
    BALANCED = "balanced"  # Model CPU offload + VAE tiling
    AGGRESSIVE = "aggressive"  # Sequential offload + all optimizations
    ULTRA = "ultra"  # Group offload + layerwise + everything


@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization."""

    # Optimization level
    level: OptimizationLevel = OptimizationLevel.BALANCED

    # Offloading strategies
    enable_sequential_offload: bool = False
    enable_model_offload: bool = True
    enable_group_offload: bool = False

    # VAE optimizations
    enable_vae_tiling: bool = True
    enable_vae_slicing: bool = True
    vae_decode_chunk_size: int = 2  # Lower = less memory

    # Attention optimizations
    enable_attention_slicing: bool = True
    attention_slice_size: int = 1  # 1 = most memory efficient
    enable_xformers: bool = True

    # UNet optimizations
    enable_forward_chunking: bool = False

    # Aggressive cleanup
    enable_immediate_gc: bool = True
    enable_cuda_cache_clear: bool = True
    clear_cache_after_each_component: bool = False

    # Quantization
    use_fp16: bool = True
    use_int8_quantization: bool = False
    use_fp8_layerwise: bool = False

    # Advanced group offload settings
    group_offload_type: str = "leaf_level"  # "leaf_level" or "block_level"
    group_offload_use_stream: bool = True
    group_offload_non_blocking: bool = True

    @classmethod
    def from_level(cls, level: OptimizationLevel) -> "MemoryOptimizationConfig":
        """Create config from optimization level."""
        if level == OptimizationLevel.NONE:
            return cls(
                level=level,
                enable_model_offload=False,
                enable_vae_tiling=False,
                enable_vae_slicing=False,
                enable_attention_slicing=False,
                enable_xformers=False,
                enable_immediate_gc=False,
                enable_cuda_cache_clear=False,
            )
        elif level == OptimizationLevel.BALANCED:
            return cls(
                level=level,
                enable_model_offload=True,
                enable_vae_tiling=True,
                enable_vae_slicing=True,
                enable_attention_slicing=True,
                enable_xformers=True,
                enable_immediate_gc=True,
                vae_decode_chunk_size=4,
            )
        elif level == OptimizationLevel.AGGRESSIVE:
            return cls(
                level=level,
                enable_sequential_offload=True,
                enable_model_offload=False,  # Sequential takes precedence
                enable_vae_tiling=True,
                enable_vae_slicing=True,
                enable_attention_slicing=True,
                enable_xformers=True,
                enable_forward_chunking=True,
                enable_immediate_gc=True,
                enable_cuda_cache_clear=True,
                clear_cache_after_each_component=True,
                vae_decode_chunk_size=2,
            )
        else:  # ULTRA
            return cls(
                level=level,
                enable_group_offload=True,
                enable_vae_tiling=True,
                enable_vae_slicing=True,
                enable_attention_slicing=True,
                enable_xformers=True,
                enable_forward_chunking=True,
                enable_immediate_gc=True,
                enable_cuda_cache_clear=True,
                clear_cache_after_each_component=True,
                vae_decode_chunk_size=1,
                use_fp8_layerwise=True,
            )


class MemoryOptimizer:
    """
    Aggressive memory optimizer that frees memory IMMEDIATELY.

    Our market differentiator: fastest memory release in the industry.

    Example:
        >>> optimizer = MemoryOptimizer(OptimizationLevel.AGGRESSIVE)
        >>> optimizer.optimize_pipeline(pipeline)
        >>> # Memory is freed immediately after each component use
    """

    def __init__(self, config: MemoryOptimizationConfig):
        """Initialize optimizer with config."""
        self.config = config
        logger.info(f"MemoryOptimizer initialized with level: {config.level.value}")

    def optimize_pipeline(self, pipeline: PipelineProtocol) -> None:
        """
        Apply ALL memory optimizations to pipeline.

        Args:
            pipeline: Diffusers pipeline to optimize
        """
        logger.info("Applying memory optimizations to pipeline...")

        # 1. Offloading strategies (mutually exclusive)
        if self.config.enable_group_offload:
            self._apply_group_offloading(pipeline)
        elif self.config.enable_sequential_offload:
            self._apply_sequential_offload(pipeline)
        elif self.config.enable_model_offload:
            self._apply_model_offload(pipeline)

        # 2. VAE optimizations
        if hasattr(pipeline, 'vae'):
            self._optimize_vae(pipeline.vae)

        # 3. Attention optimizations
        self._optimize_attention(pipeline)

        # 4. UNet optimizations
        if hasattr(pipeline, 'unet'):
            self._optimize_unet(pipeline.unet)

        # 5. Transformer optimizations (for FLUX, etc.)
        if hasattr(pipeline, 'transformer'):
            self._optimize_transformer(pipeline.transformer)

        # 6. Initial cleanup
        self._immediate_cleanup()

        logger.info("✅ All memory optimizations applied")
        self._log_memory_stats()

    def _apply_group_offloading(self, pipeline: PipelineProtocol) -> None:
        """Apply group offloading (ULTRA level)."""
        try:
            from diffusers.hooks import apply_group_offloading

            onload_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            offload_device = torch.device("cpu")

            # Apply to all major components
            components = []
            if hasattr(pipeline, 'transformer'):
                components.append(('transformer', pipeline.transformer))
            if hasattr(pipeline, 'unet'):
                components.append(('unet', pipeline.unet))
            if hasattr(pipeline, 'vae'):
                components.append(('vae', pipeline.vae))
            if hasattr(pipeline, 'text_encoder'):
                components.append(('text_encoder', pipeline.text_encoder))
            if hasattr(pipeline, 'text_encoder_2'):
                components.append(('text_encoder_2', pipeline.text_encoder_2))

            for name, component in components:
                apply_group_offloading(
                    component,
                    onload_device=onload_device,
                    offload_device=offload_device,
                    offload_type=self.config.group_offload_type,
                    use_stream=self.config.group_offload_use_stream,
                    non_blocking=self.config.group_offload_non_blocking,
                )
                logger.info(f"  ✓ Group offload applied to {name}")

                if self.config.clear_cache_after_each_component:
                    self._immediate_cleanup()

        except ImportError:
            logger.warning("Group offloading not available, falling back to model offload")
            self._apply_model_offload(pipeline)

    def _apply_sequential_offload(self, pipeline: PipelineProtocol) -> None:
        """Apply sequential CPU offload (AGGRESSIVE level)."""
        try:
            pipeline.enable_sequential_cpu_offload()
            logger.info("  ✓ Sequential CPU offload enabled")
        except Exception as e:
            logger.warning(f"Sequential offload failed: {e}")

    def _apply_model_offload(self, pipeline: PipelineProtocol) -> None:
        """Apply model CPU offload (BALANCED level)."""
        try:
            pipeline.enable_model_cpu_offload()
            logger.info("  ✓ Model CPU offload enabled")
        except Exception as e:
            logger.warning(f"Model offload failed: {e}")

    def _optimize_vae(self, vae: VAEProtocol) -> None:
        """Optimize VAE for minimum memory."""
        if self.config.enable_vae_tiling:
            try:
                vae.enable_tiling()
                logger.info("  ✓ VAE tiling enabled")
            except Exception as e:
                logger.warning(f"VAE tiling failed: {e}")

        if self.config.enable_vae_slicing:
            try:
                vae.enable_slicing()
                logger.info("  ✓ VAE slicing enabled")
            except Exception as e:
                logger.warning(f"VAE slicing failed: {e}")

        # FP8 layerwise casting (ULTRA level)
        if self.config.use_fp8_layerwise:
            try:
                vae.enable_layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn,
                    compute_dtype=torch.bfloat16
                )
                logger.info("  ✓ VAE FP8 layerwise casting enabled")
            except Exception as e:
                logger.warning(f"FP8 casting failed: {e}")

    def _optimize_attention(self, pipeline: PipelineProtocol) -> None:
        """Optimize attention mechanism."""
        if self.config.enable_attention_slicing:
            try:
                pipeline.enable_attention_slicing(self.config.attention_slice_size)
                logger.info(f"  ✓ Attention slicing enabled (size: {self.config.attention_slice_size})")
            except Exception as e:
                logger.warning(f"Attention slicing failed: {e}")

        if self.config.enable_xformers:
            try:
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("  ✓ xFormers memory efficient attention enabled")
            except Exception as e:
                logger.warning(f"xFormers failed (not installed?): {e}")

    def _optimize_unet(self, unet: UNetProtocol) -> None:
        """Optimize UNet."""
        if self.config.enable_forward_chunking:
            try:
                unet.enable_forward_chunking()
                logger.info("  ✓ UNet forward chunking enabled")
            except Exception as e:
                logger.warning(f"Forward chunking failed: {e}")

    def _optimize_transformer(self, transformer: TransformerProtocol) -> None:
        """Optimize transformer (for FLUX, etc.)."""
        # FP8 layerwise casting (ULTRA level)
        if self.config.use_fp8_layerwise:
            try:
                transformer.enable_layerwise_casting(
                    storage_dtype=torch.float8_e4m3fn,
                    compute_dtype=torch.bfloat16
                )
                logger.info("  ✓ Transformer FP8 layerwise casting enabled")
            except Exception as e:
                logger.warning(f"FP8 casting failed: {e}")

    def _immediate_cleanup(self) -> None:
        """FREE MEMORY IMMEDIATELY - our market differentiator."""
        if self.config.enable_immediate_gc:
            gc.collect()

        if self.config.enable_cuda_cache_clear and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def cleanup_after_generation(self) -> None:
        """Clean up after image generation."""
        self._immediate_cleanup()
        logger.debug("Memory cleaned after generation")

    def cleanup_after_model_load(self) -> None:
        """Clean up after loading a model."""
        if self.config.clear_cache_after_each_component:
            self._immediate_cleanup()
            logger.debug("Memory cleaned after model load")

    def cleanup_after_model_unload(self) -> None:
        """Clean up after unloading a model - AGGRESSIVE."""
        # Always clean aggressively when unloading
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.debug("Memory aggressively cleaned after model unload")

    def _log_memory_stats(self) -> None:
        """Log current memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def get_memory_stats(self) -> MemoryStatistics:
        """Get current memory statistics."""
        if not torch.cuda.is_available():
            return MemoryStatistics(
                allocated_gb=0.0, reserved_gb=0.0, free_gb=0.0, total_gb=0.0
            )

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - allocated

        return MemoryStatistics(
            allocated_gb=round(allocated, 2),
            reserved_gb=round(reserved, 2),
            free_gb=round(free, 2),
            total_gb=round(total, 2),
        )


class MemoryMonitor:
    """Monitor memory usage during generation."""

    def __init__(self, optimizer: MemoryOptimizer):
        """Initialize monitor."""
        self.optimizer = optimizer
        self.peak_memory = 0.0
        self.start_memory = 0.0

    def __enter__(self):
        """Start monitoring."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated() / 1024**3
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and cleanup."""
        if torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"Peak memory: {self.peak_memory:.2f}GB (delta: {self.peak_memory - self.start_memory:.2f}GB)")

        # IMMEDIATE CLEANUP
        self.optimizer.cleanup_after_generation()

    def get_peak_memory(self) -> float:
        """Get peak memory usage in GB."""
        return self.peak_memory
