"""Configuration loading utilities."""

import json
import os
from pathlib import Path

import yaml
from infrastructure.config.types.config_data_map import ConfigDataMap
from infrastructure.config.types.raw_config_data import RawConfigData
from infrastructure.errors.configuration_error import ConfigurationError
from infrastructure.serialization.protocol_serializer import ProtocolSerializer


class ConfigLoader:
    """Utility class for loading configurations from various sources."""

    @staticmethod
    def load_from_env(prefix: str) -> ConfigDataMap:
        """Load configuration from environment variables with prefix.

        Args:
            prefix: Environment variable prefix (e.g., 'LLM_', 'DB_').

        Returns:
            Type-safe configuration values.
        """
        temp_config = ProtocolSerializer.serialize_env_config({})
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix) :].lower()
                temp_config[config_key] = value
        return ConfigDataMap.from_dict(temp_config)

    @staticmethod
    def load_from_yaml(file_path: Path) -> RawConfigData:
        """Load configuration from YAML file.

        Args:
            file_path: Path to YAML configuration file.

        Returns:
            Type-safe configuration data.

        Raises:
            ConfigurationError: If file cannot be read or parsed.
        """
        try:
            with open(file_path, encoding="utf-8") as file:
                data = yaml.safe_load(file)
                if data is None:
                    return RawConfigData.empty()
                return RawConfigData.from_dict(data)
        except FileNotFoundError as e:
            raise ConfigurationError(
                f"Configuration file not found: {file_path}"
            ) from e
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {file_path}: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Error loading {file_path}: {e}") from e

    @staticmethod
    def load_from_json(file_path: Path) -> RawConfigData:
        """Load configuration from JSON file.

        Args:
            file_path: Path to JSON configuration file.

        Returns:
            Type-safe configuration data.

        Raises:
            ConfigurationError: If file cannot be read or parsed.
        """
        try:
            with open(file_path, encoding="utf-8") as file:
                data = json.load(file)
                return RawConfigData.from_dict(data)
        except FileNotFoundError as e:
            raise ConfigurationError(
                f"Configuration file not found: {file_path}"
            ) from e
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {file_path}: {e}") from e
        except Exception as e:
            raise ConfigurationError(f"Error loading {file_path}: {e}") from e

    @staticmethod
    def get_env_var(key: str, default: str | None = None, required: bool = True) -> str:
        """Get environment variable with validation.

        Args:
            key: Environment variable name.
            default: Default value if not found.
            required: Whether the variable is required.

        Returns:
            Environment variable value.

        Raises:
            ConfigurationError: If required variable is missing.
        """
        value = os.getenv(key, default)
        if required and value is None:
            raise ConfigurationError(
                f"Required environment variable '{key}' is not set"
            )
        return value or ""

    @staticmethod
    def get_env_bool(key: str, default: bool = False) -> bool:
        """Get boolean environment variable.

        Args:
            key: Environment variable name.
            default: Default value if not found.

        Returns:
            Boolean value.
        """
        value = os.getenv(key, "").lower()
        if value in ("true", "1", "yes", "on"):
            return True
        if value in ("false", "0", "no", "off"):
            return False
        return default

    @staticmethod
    def get_env_int(key: str, default: int | None = None, required: bool = True) -> int:
        """Get integer environment variable.

        Args:
            key: Environment variable name.
            default: Default value if not found.
            required: Whether the variable is required.

        Returns:
            Integer value.

        Raises:
            ConfigurationError: If variable is invalid or missing when required.
        """
        value = os.getenv(key)
        if value is None:
            if required and default is None:
                raise ConfigurationError(
                    f"Required environment variable '{key}' is not set"
                )
            return default or 0

        try:
            return int(value)
        except ValueError as e:
            raise ConfigurationError(
                f"Environment variable '{key}' must be an integer, got: {value}"
            ) from e
