"""Base configuration class for all infrastructure configurations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from infrastructure.config.types.base_config_data import BaseConfigData


class BaseInfrastructureConfig(ABC):
    """Base class for all infrastructure configurations.

    Enforces consistent validation and loading patterns across
    all configuration classes in the infrastructure layer.
    """

    @classmethod
    @abstractmethod
    def from_environment(cls) -> BaseInfrastructureConfig:
        """Load configuration from environment variables.

        Returns:
            Configured instance loaded from environment.

        Raises:
            ConfigurationError: If required environment variables are missing.
        """

    @classmethod
    @abstractmethod
    def from_config_data(cls, data: BaseConfigData) -> BaseInfrastructureConfig:
        """Load configuration from dictionary.

        Args:
            data: Configuration data dictionary.

        Returns:
            Configured instance loaded from dictionary.

        Raises:
            ConfigurationError: If required fields are missing or invalid.
        """

    @abstractmethod
    def validate(self) -> list[str]:
        """Validate configuration and return any errors.

        Returns:
            List of validation error messages. Empty list if valid.
        """

    @abstractmethod
    def to_config_data(self) -> BaseConfigData:
        """Convert configuration to typed data.

        Returns:
            Typed configuration data.
        """

    def is_valid(self) -> bool:
        """Check if configuration is valid.

        Returns:
            True if configuration passes validation, False otherwise.
        """
        return len(self.validate()) == 0

    def get_validation_summary(self) -> str:
        """Get human-readable validation summary.

        Returns:
            Validation summary string.
        """
        errors = self.validate()
        if not errors:
            return "✅ Configuration is valid"

        return f"❌ Configuration has {len(errors)} errors:\n" + "\n".join(
            f"  - {error}" for error in errors
        )
