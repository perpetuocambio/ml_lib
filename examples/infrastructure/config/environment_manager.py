"""Manager for environment-specific configuration loading."""

import os

from infrastructure.serialization.protocol_serializer import ProtocolSerializer


class EnvironmentManager:
    """Manager for environment-specific configuration loading."""

    @staticmethod
    def get_current_environment() -> str:
        """Get current environment from environment variables.

        Returns:
            Environment name (development, staging, production).
        """
        return os.getenv("PYINTELCIVIL_ENV", "development").lower()

    @staticmethod
    def get_environment_config_path(environment: str) -> str | None:
        """Get configuration file path for environment.

        Args:
            environment: Environment name.

        Returns:
            Configuration file path or None if not found.
        """
        config_data = ProtocolSerializer.serialize_environment_mapping(
            {
                "development": "config/environments/development.yaml",
                "staging": "config/environments/staging.yaml",
                "production": "config/environments/production.yaml",
            }
        )
        return config_data.get(environment)
