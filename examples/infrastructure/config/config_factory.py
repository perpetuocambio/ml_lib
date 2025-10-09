"""Advanced centralized configuration factory for all infrastructure components."""

from __future__ import annotations

import logging
import threading
from typing import TypeVar

from infrastructure.config.base.base_config import BaseInfrastructureConfig
from infrastructure.config.config_registry import ConfigRegistry
from infrastructure.config.configuration_cache import ConfigurationCache
from infrastructure.config.environment_manager import EnvironmentManager
from infrastructure.serialization.protocol_serializer import ProtocolSerializer

# Import optional configuration classes - graceful fallback if not found
try:
    from infrastructure.config.providers.llm_provider_config import LLMProviderConfig
except ImportError:
    LLMProviderConfig = None

try:
    from infrastructure.config.providers.database_provider_config import (
        DatabaseProviderConfig,
    )
except ImportError:
    DatabaseProviderConfig = None

try:
    from infrastructure.config.algorithms.scraping_config import ScrapingConfig
except ImportError:
    ScrapingConfig = None

try:
    from infrastructure.config.environments.development import DevelopmentConfig
    from infrastructure.config.environments.production import ProductionConfig
    from infrastructure.config.environments.staging import StagingConfig
except ImportError:
    DevelopmentConfig = None
    ProductionConfig = None
    StagingConfig = None

try:
    from infrastructure.config.templates.config_presets import (
        ConfigPresets,
        ConfigTemplates,
    )
except ImportError:
    ConfigPresets = None
    ConfigTemplates = None

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseInfrastructureConfig)


class ConfigFactory:
    """Advanced singleton factory for all infrastructure configurations.

    Features:
    - Thread-safe singleton pattern
    - Configuration caching with TTL
    - Hot reload capabilities
    - Environment-specific loading
    - Dynamic registration
    - Validation pipeline
    - Health monitoring
    """

    _instance: ConfigFactory | None = None
    _lock = threading.RLock()

    def __init__(self):
        """Initialize factory components."""
        self._cache = ConfigurationCache(ttl_seconds=3600)
        self._environment = EnvironmentManager.get_current_environment()
        self._health_checks_enabled = True
        self._validation_strict = True

        # Register default configuration types
        self._register_default_configs()

    def __new__(cls) -> ConfigFactory:
        """Thread-safe singleton implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def _register_default_configs(self) -> None:
        """Register default configuration types."""
        # LLM Provider Config
        if LLMProviderConfig is not None:
            ConfigRegistry.register("llm", LLMProviderConfig)

        # Database Provider Config
        if DatabaseProviderConfig is not None:
            ConfigRegistry.register("database", DatabaseProviderConfig)

        # Scraping Algorithm Config
        if ScrapingConfig is not None:
            ConfigRegistry.register("scraping", ScrapingConfig)

        # Environment Configs
        if DevelopmentConfig is not None:
            ConfigRegistry.register("development", DevelopmentConfig)
        if StagingConfig is not None:
            ConfigRegistry.register("staging", StagingConfig)
        if ProductionConfig is not None:
            ConfigRegistry.register("production", ProductionConfig)

    def get_config(
        self, config_type: str, force_reload: bool = False
    ) -> BaseInfrastructureConfig:
        """Get configuration by type with advanced caching.

        Args:
            config_type: Configuration type identifier.
            force_reload: Force reload from environment.

        Returns:
            Configuration instance.

        Raises:
            ValueError: If configuration type is not registered.
            RuntimeError: If configuration fails validation in strict mode.
        """
        with self._lock:
            # Check cache first
            if not force_reload:
                cached_config = self._cache.get(config_type)
                if cached_config is not None:
                    logger.debug(f"Retrieved {config_type} config from cache")
                    return cached_config

            # Load configuration
            config = self._load_config(config_type)

            # Validate if strict mode enabled
            if self._validation_strict:
                errors = config.validate()
                if errors:
                    error_msg = (
                        f"Configuration validation failed for {config_type}: {errors}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)

            # Cache the configuration
            self._cache.put(config_type, config)
            logger.info(f"Loaded and cached {config_type} configuration")

            return config

    def _load_config(self, config_type: str) -> BaseInfrastructureConfig:
        """Load configuration from environment.

        Args:
            config_type: Configuration type identifier.

        Returns:
            Configuration instance.

        Raises:
            ValueError: If configuration type is not registered.
        """
        config_class = ConfigRegistry.get_config_class(config_type)
        if config_class is None:
            raise ValueError(f"Unknown configuration type: {config_type}")

        try:
            return config_class.from_environment()
        except Exception as e:
            logger.error(f"Failed to load {config_type} configuration: {e}")
            raise

    # Convenience methods for specific configurations
    def get_llm_config(self, force_reload: bool = False) -> BaseInfrastructureConfig:
        """Get LLM provider configuration."""
        return self.get_config("llm", force_reload)

    def get_database_config(
        self, force_reload: bool = False
    ) -> BaseInfrastructureConfig:
        """Get database provider configuration."""
        return self.get_config("database", force_reload)

    def get_scraping_config(
        self, force_reload: bool = False
    ) -> BaseInfrastructureConfig:
        """Get scraping algorithm configuration."""
        return self.get_config("scraping", force_reload)

    def get_environment_config(
        self, force_reload: bool = False
    ) -> BaseInfrastructureConfig:
        """Get environment-specific configuration."""
        return self.get_config(self._environment, force_reload)

    # Cache management
    def reload_config(self, config_type: str) -> BaseInfrastructureConfig:
        """Force reload specific configuration.

        Args:
            config_type: Configuration type identifier.

        Returns:
            Reloaded configuration instance.
        """
        logger.info(f"Force reloading {config_type} configuration")
        return self.get_config(config_type, force_reload=True)

    def reload_all_configs(self) -> list[BaseInfrastructureConfig]:
        """Force reload all cached configurations.

        Returns:
            List of all reloaded configurations.
        """
        logger.info("Force reloading all configurations")

        # Get all currently cached types
        cache_info = self._cache.get_cache_info()
        cached_types = cache_info.cached_types

        # Clear cache
        self._cache.clear()

        # Reload all previously cached configurations
        reloaded = []
        for config_type in cached_types:
            try:
                config = self.get_config(config_type)
                reloaded.append(config)
            except Exception as e:
                logger.error(f"Failed to reload {config_type}: {e}")

        return reloaded

    def evict_config(self, config_type: str) -> None:
        """Remove configuration from cache.

        Args:
            config_type: Configuration type identifier.
        """
        self._cache.evict(config_type)
        logger.debug(f"Evicted {config_type} from cache")

    def clear_cache(self) -> None:
        """Clear all cached configurations."""
        self._cache.clear()
        logger.info("Cleared all configuration cache")

    # Validation and health checks
    def validate_all_configs(self) -> list[str]:
        """Validate all loaded configurations.

        Returns:
            List of validation errors.
        """
        validation_errors = []
        cache_info = self._cache.get_cache_info()

        for config_type in cache_info.cached_types:
            try:
                config = self._cache.get(config_type)
                if config:
                    errors = config.validate()
                    if errors:
                        validation_errors.extend(
                            [f"{config_type}: {error}" for error in errors]
                        )
            except Exception as e:
                validation_errors.append(f"{config_type}: Validation error: {str(e)}")

        return validation_errors

    def health_check(self) -> list[str]:
        """Perform health check on configuration system.

        Returns:
            List of health check issues.
        """
        health_issues = []

        if self._health_checks_enabled:
            validation_errors = self.validate_all_configs()
            health_issues.extend(validation_errors)

        return health_issues

    # Configuration management
    def set_config(self, config_type: str, config: BaseInfrastructureConfig) -> None:
        """Set configuration instance (mainly for testing).

        Args:
            config_type: Configuration type identifier.
            config: Configuration instance.
        """
        self._cache.put(config_type, config)
        logger.debug(f"Set {config_type} configuration manually")

    def get_config_summary(self) -> str:
        """Get human-readable configuration summary.

        Returns:
            Configuration summary string.
        """
        cache_info = self._cache.get_cache_info()
        cached_types = cache_info.cached_types

        if not cached_types:
            return f"🏗️ ConfigFactory [{self._environment}]: No configurations loaded"

        summary_lines = [
            f"🏗️ ConfigFactory [{self._environment}]: {len(cached_types)} configurations loaded:",
            f"  📦 Cache TTL: {cache_info.ttl_seconds}s",
        ]

        validation_errors = self.validate_all_configs()
        error_count = len(validation_errors)
        for config_type in cached_types:
            status = "✅ Valid" if error_count == 0 else "❌ Has errors"
            summary_lines.append(f"  - {config_type}: {status}")

        return "\n".join(summary_lines)

    # Environment and settings
    def get_environment(self) -> str:
        """Get current environment.

        Returns:
            Environment name.
        """
        return self._environment

    def set_validation_strict(self, strict: bool) -> None:
        """Enable/disable strict validation mode.

        Args:
            strict: Whether to enable strict validation.
        """
        self._validation_strict = strict
        logger.info(f"Set validation strict mode: {strict}")

    def set_health_checks(self, enabled: bool) -> None:
        """Enable/disable health checks.

        Args:
            enabled: Whether to enable health checks.
        """
        self._health_checks_enabled = enabled
        logger.info(f"Set health checks enabled: {enabled}")

    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """Set cache TTL.

        Args:
            ttl_seconds: Time to live in seconds.
        """
        self._cache.ttl_seconds = ttl_seconds
        logger.info(f"Set cache TTL: {ttl_seconds}s")

    # Template and preset methods
    def create_config_from_preset(
        self, config_type: str, preset_name: str
    ) -> BaseInfrastructureConfig:
        """Create configuration instance from preset.

        Args:
            config_type: Type of configuration.
            preset_name: Name of the preset.

        Returns:
            Configuration instance.
        """
        if ConfigPresets is None:
            raise RuntimeError("Config presets not available")

        preset_data = None
        if config_type == "llm":
            preset_data = ConfigPresets.get_llm_preset(preset_name)
            config_class = ConfigRegistry.get_config_class("llm")
        elif config_type == "scraping":
            preset_data = ConfigPresets.get_scraping_preset(preset_name)
            config_class = ConfigRegistry.get_config_class("scraping")
        else:
            raise ValueError(
                f"Preset creation not supported for config type: {config_type}"
            )

        if config_class is None:
            raise ValueError(f"Configuration class not found for type: {config_type}")

        serialized_data = ProtocolSerializer.serialize_config_data(preset_data)
        return config_class.from_dict(serialized_data)

    def load_template(self, template_name: str) -> list[BaseInfrastructureConfig]:
        """Load complete configuration template.

        Args:
            template_name: Name of the template (development, production, research).

        Returns:
            List of configuration instances.
        """
        if ConfigTemplates is None:
            raise RuntimeError("Config templates not available")

        if template_name == "development":
            template_data = ConfigTemplates.generate_development_template()
        elif template_name == "production":
            template_data = ConfigTemplates.generate_production_template()
        elif template_name == "research":
            template_data = ConfigTemplates.generate_research_template()
        else:
            raise ValueError(f"Unknown template: {template_name}")

        configs = []
        serialized_template = ProtocolSerializer.serialize_template_data(template_data)
        for config_type, config_data in serialized_template.items():
            if config_type == "environment":
                continue  # Skip environment settings for now

            config_class = ConfigRegistry.get_config_class(config_type)
            if config_class:
                serialized_config = ProtocolSerializer.serialize_config_data(
                    config_data
                )
                config = config_class.from_dict(serialized_config)
                configs.append(config)
                # Cache the configuration
                self._cache.put(config_type, config)

        return configs

    def get_available_presets(self) -> list[str]:
        """Get all available configuration presets.

        Returns:
            List of preset names.
        """
        if ConfigPresets is None:
            raise RuntimeError("Config presets not available")

        all_presets = ConfigPresets.get_all_presets()
        preset_names = []
        for config_type, presets in all_presets.items():
            preset_names.extend([f"{config_type}:{preset}" for preset in presets])
        return preset_names

    # Legacy compatibility methods
    @classmethod
    def reset_cache(cls) -> None:
        """Legacy method: Reset all cached configurations."""
        if cls._instance:
            cls._instance.clear_cache()
