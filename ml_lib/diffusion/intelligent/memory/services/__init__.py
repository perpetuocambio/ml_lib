"""Memory optimization services - DEPRECATED: Use ml_lib.diffusion.services instead."""

# Re-export from new location for backward compatibility
from ml_lib.diffusion.services.memory_optimizer import (
    MemoryOptimizer,
    MemoryOptimizationConfig,
    OptimizationLevel,
    MemoryMonitor,
)

__all__ = [
    "MemoryOptimizer",
    "MemoryOptimizationConfig",
    "OptimizationLevel",
    "MemoryMonitor",
]
