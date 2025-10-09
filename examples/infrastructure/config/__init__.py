"""Infrastructure configuration module.

Centralized configuration management for all infrastructure components.
Provides a unified interface to access all configurations through ConfigFactory.
"""

from pathlib import Path

from infrastructure.config.base import (
    BaseInfrastructureConfig,
    ConfigLoader,
    ConfigValidator,
)
from infrastructure.config.config_factory import ConfigFactory

# Re-export main configuration classes for convenience
# Note: Lazy imports to avoid circular dependencies during migration
# from infrastructure.config.algorithms import ExtractionConfig, ScrapingConfig
# from infrastructure.config.providers import DatabaseProviderConfig, LLMProviderConfig

__all__ = [
    # Core configuration infrastructure
    "BaseInfrastructureConfig",
    "ConfigFactory",
    "ConfigLoader",
    "ConfigValidator",
    # Provider configurations will be available via ConfigFactory during migration
    # "LLMProviderConfig",
    # "DatabaseProviderConfig",
    # Algorithm configurations will be available via ConfigFactory during migration
    # "ExtractionConfig",
    # "ScrapingConfig",
]


def get_config_path(config_type: str) -> str:
    """Get the configuration path for a specific config type.

    Args:
        config_type: Type of configuration ('providers', 'algorithms', 'application', etc.)

    Returns:
        Path to the configuration directory
    """
    base_config_path = Path(__file__).parent
    return str(base_config_path / config_type)
