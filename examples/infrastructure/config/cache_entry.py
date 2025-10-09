"""Cache entry data structure."""

from dataclasses import dataclass
from datetime import datetime

from infrastructure.config.base.base_config import BaseInfrastructureConfig


@dataclass
class CacheEntry:
    """Cache entry data structure."""

    config: BaseInfrastructureConfig
    timestamp: datetime
