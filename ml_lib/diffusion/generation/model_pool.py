"""Model pool with intelligent caching and eviction."""

import gc
import logging
import time
from collections import defaultdict
from typing import Callable, Optional, TypeVar

from ml_lib.diffusion.models import (
    LoadedModel,
    EvictionPolicy,
)
from ml_lib.diffusion.models.value_objects import (
    PoolStatistics,
    ModelProtocol,
)

logger = logging.getLogger(__name__)

# Generic type for models
T = TypeVar('T', bound=ModelProtocol)


class ModelPool:
    """Pool of loaded models with LRU eviction."""

    def __init__(
        self,
        max_size_gb: float = 20.0,
        eviction_policy: EvictionPolicy = EvictionPolicy.LRU,
    ):
        """
        Initialize model pool.

        Args:
            max_size_gb: Maximum pool size in GB
            eviction_policy: Policy for evicting models
        """
        self.max_size_gb = max_size_gb
        self.eviction_policy = eviction_policy

        # Storage - keep internal dicts for performance
        # but don't expose them in public API
        self._loaded_models: dict[str, LoadedModel] = {}
        self._access_times: dict[str, float] = {}
        self._access_counts: dict[str, int] = defaultdict(int)

        logger.info(
            f"ModelPool initialized: max_size={max_size_gb}GB, "
            f"policy={eviction_policy.value}"
        )

    def load(
        self,
        model_id: str,
        loader_fn: Callable[[], T],
        estimated_size_gb: float = 2.0,
    ) -> T:
        """
        Load model with automatic eviction if needed.

        Args:
            model_id: Unique model identifier
            loader_fn: Function to load the model (takes no args)
            estimated_size_gb: Estimated model size

        Returns:
            Loaded model
        """
        # Check if already loaded
        if model_id in self._loaded_models:
            logger.debug(f"Model {model_id} already loaded (cache hit)")
            self._update_access(model_id)
            return self._loaded_models[model_id].model  # type: ignore

        # Evict if necessary
        while self._current_size() + estimated_size_gb > self.max_size_gb:
            if not self._evict_one():
                logger.warning("Cannot evict more models, pool may exceed limit")
                break

        # Load model
        logger.info(f"Loading model {model_id} ({estimated_size_gb:.2f}GB)")
        start_time = time.time()

        try:
            model = loader_fn()
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

        load_time = time.time() - start_time

        # Store in pool
        loaded_model = LoadedModel(
            model=model,
            size_gb=estimated_size_gb,
            loaded_at=time.time(),
        )

        self._loaded_models[model_id] = loaded_model
        self._update_access(model_id)

        logger.info(
            f"Model {model_id} loaded in {load_time:.2f}s "
            f"(pool: {self._current_size():.2f}/{self.max_size_gb:.2f}GB)"
        )

        return model  # type: ignore

    def unload(self, model_id: str) -> bool:
        """
        Unload model from pool.

        Args:
            model_id: Model to unload

        Returns:
            True if unloaded
        """
        if model_id not in self._loaded_models:
            return False

        size = self._loaded_models[model_id].size_gb

        # Remove from tracking
        del self._loaded_models[model_id]
        self._access_times.pop(model_id, None)
        self._access_counts.pop(model_id, None)

        # Force cleanup
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            logger.debug("torch not available, skipping CUDA cache clearing")

        logger.info(f"Unloaded model {model_id} ({size:.2f}GB)")
        return True

    def is_loaded(self, model_id: str) -> bool:
        """
        Check if model is loaded.

        Args:
            model_id: Model ID

        Returns:
            True if loaded
        """
        return model_id in self._loaded_models

    def get_model(self, model_id: str) -> Optional[T]:
        """
        Get model from pool.

        Args:
            model_id: Model ID

        Returns:
            Model or None
        """
        if model_id in self._loaded_models:
            self._update_access(model_id)
            return self._loaded_models[model_id].model  # type: ignore
        return None

    def clear(self) -> None:
        """Clear all models from pool."""
        model_ids = list(self._loaded_models.keys())
        for model_id in model_ids:
            self.unload(model_id)

        logger.info("Pool cleared")

    def get_stats(self) -> PoolStatistics:
        """
        Get pool statistics.

        Returns:
            PoolStatistics object
        """
        current_size = self._current_size()
        utilization = current_size / self.max_size_gb if self.max_size_gb > 0 else 0.0

        return PoolStatistics(
            loaded_count=len(self._loaded_models),
            current_size_gb=current_size,
            max_size_gb=self.max_size_gb,
            utilization=utilization,
            model_ids=list(self._loaded_models.keys()),
        )

    def _current_size(self) -> float:
        """Get current pool size in GB."""
        return sum(m.size_gb for m in self._loaded_models.values())

    def _update_access(self, model_id: str) -> None:
        """Update access tracking for a model."""
        self._access_times[model_id] = time.time()
        self._access_counts[model_id] += 1

        # Update LoadedModel tracking
        if model_id in self._loaded_models:
            self._loaded_models[model_id].update_access()

    def _evict_one(self) -> bool:
        """
        Evict one model based on policy.

        Returns:
            True if a model was evicted
        """
        if not self._loaded_models:
            return False

        # Select model to evict based on policy
        if self.eviction_policy == EvictionPolicy.LRU:
            victim = self._select_lru()
        elif self.eviction_policy == EvictionPolicy.LFU:
            victim = self._select_lfu()
        elif self.eviction_policy == EvictionPolicy.SIZE:
            victim = self._select_largest()
        elif self.eviction_policy == EvictionPolicy.FIFO:
            victim = self._select_fifo()
        else:
            victim = self._select_lru()  # Default

        if victim:
            logger.debug(f"Evicting model {victim} ({self.eviction_policy.value} policy)")
            self.unload(victim)
            return True

        return False

    def _select_lru(self) -> Optional[str]:
        """Select least recently used model."""
        if not self._access_times:
            return None

        return min(self._access_times.items(), key=lambda x: x[1])[0]

    def _select_lfu(self) -> Optional[str]:
        """Select least frequently used model."""
        if not self._access_counts:
            return None

        return min(self._access_counts.items(), key=lambda x: x[1])[0]

    def _select_largest(self) -> Optional[str]:
        """Select largest model."""
        if not self._loaded_models:
            return None

        return max(self._loaded_models.items(), key=lambda x: x[1].size_gb)[0]

    def _select_fifo(self) -> Optional[str]:
        """Select first in (oldest loaded)."""
        if not self._loaded_models:
            return None

        return min(
            self._loaded_models.items(),
            key=lambda x: x[1].loaded_at,
        )[0]
