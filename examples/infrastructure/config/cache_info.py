"""Cache information data structure."""

from dataclasses import dataclass


@dataclass
class CacheInfo:
    """Cache statistics data structure."""

    cache_size: int
    cached_types: list[str]
    timestamps: list[str]
    ttl_seconds: int
