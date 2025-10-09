"""Thread-safe configuration cache with expiration and hot reload."""

import threading
from datetime import datetime

from infrastructure.config.base.base_config import BaseInfrastructureConfig
from infrastructure.config.cache_entry import CacheEntry
from infrastructure.config.cache_info import CacheInfo


class ConfigurationCache:
    """Thread-safe configuration cache with expiration and hot reload."""

    def __init__(self, ttl_seconds: int = 3600):
        """Initialize cache.

        Args:
            ttl_seconds: Time to live for cached configurations.
        """
        self._cache_entries: list[CacheEntry] = []
        self._lock = threading.RLock()
        self.ttl_seconds = ttl_seconds

    def get(self, config_type: str) -> BaseInfrastructureConfig | None:
        """Get configuration from cache.

        Args:
            config_type: Configuration type identifier.

        Returns:
            Cached configuration or None if not found/expired.
        """
        with self._lock:
            entry = self._find_entry(config_type)
            if entry is None:
                return None

            # Check expiration
            if self._is_expired(entry):
                self._evict(config_type)
                return None

            return entry.config

    def put(self, config_type: str, config: BaseInfrastructureConfig) -> None:
        """Store configuration in cache.

        Args:
            config_type: Configuration type identifier.
            config: Configuration instance.
        """
        with self._lock:
            # Remove existing entry if present
            self._cache_entries = [
                e
                for e in self._cache_entries
                if e.config.__class__.__name__ != config_type
            ]
            # Add new entry
            self._cache_entries.append(CacheEntry(config, datetime.now()))

    def evict(self, config_type: str) -> None:
        """Evict configuration from cache.

        Args:
            config_type: Configuration type identifier.
        """
        with self._lock:
            self._evict(config_type)

    def clear(self) -> None:
        """Clear all cached configurations."""
        with self._lock:
            self._cache_entries.clear()

    def get_cache_info(self) -> CacheInfo:
        """Get cache statistics.

        Returns:
            Cache information data structure.
        """
        with self._lock:
            return CacheInfo(
                cache_size=len(self._cache_entries),
                cached_types=[e.config.__class__.__name__ for e in self._cache_entries],
                timestamps=[e.timestamp.isoformat() for e in self._cache_entries],
                ttl_seconds=self.ttl_seconds,
            )

    def _find_entry(self, config_type: str) -> CacheEntry | None:
        """Find cache entry by config type."""
        for entry in self._cache_entries:
            if entry.config.__class__.__name__ == config_type:
                return entry
        return None

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if configuration is expired."""
        age = (datetime.now() - entry.timestamp).total_seconds()
        return age > self.ttl_seconds

    def _evict(self, config_type: str) -> None:
        """Remove configuration from cache."""
        self._cache_entries = [
            e for e in self._cache_entries if e.config.__class__.__name__ != config_type
        ]
