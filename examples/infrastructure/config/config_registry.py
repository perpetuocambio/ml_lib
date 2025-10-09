"""Registry of configuration types for dynamic loading."""

import logging

from infrastructure.config.base.base_config import BaseInfrastructureConfig
from infrastructure.config.config_registry_data import ConfigRegistryData

logger = logging.getLogger(__name__)


class ConfigRegistry:
    """Registry of configuration types for dynamic loading."""

    _registry: list[ConfigRegistryData] = []

    @classmethod
    def register(
        cls, config_type: str, config_class: type[BaseInfrastructureConfig]
    ) -> None:
        """Register a configuration class.

        Args:
            config_type: Configuration type identifier.
            config_class: Configuration class.
        """
        # Remove existing entry if present
        cls._registry = [
            entry for entry in cls._registry if entry.config_type != config_type
        ]
        # Add new entry
        cls._registry.append(ConfigRegistryData(config_type, config_class))
        logger.debug(f"Registered config type: {config_type}")

    @classmethod
    def get_config_class(
        cls, config_type: str
    ) -> type[BaseInfrastructureConfig] | None:
        """Get configuration class by type.

        Args:
            config_type: Configuration type identifier.

        Returns:
            Configuration class or None if not found.
        """
        for entry in cls._registry:
            if entry.config_type == config_type:
                return entry.config_class
        return None

    @classmethod
    def get_all_types(cls) -> list[str]:
        """Get all registered configuration types.

        Returns:
            List of configuration type identifiers.
        """
        return [entry.config_type for entry in cls._registry]
