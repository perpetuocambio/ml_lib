"""Database provider configuration."""

from __future__ import annotations

import re
from dataclasses import dataclass

from infrastructure.config.base.base_config import BaseInfrastructureConfig
from infrastructure.config.base.config_loader import ConfigLoader
from infrastructure.config.base.config_validator import ConfigValidator
from infrastructure.config.providers.database_config_data import DatabaseConfigData
from infrastructure.persistence.database.enums.database_type import DatabaseType


@dataclass(frozen=True)
class DatabaseProviderConfig(BaseInfrastructureConfig):
    """Configuration for database providers.

    Centralizes all database configuration with proper validation
    and environment loading capabilities.
    """

    database_type: DatabaseType
    connection_string: str
    min_pool_size: int = 5
    max_pool_size: int = 20
    command_timeout: int = 60
    connection_timeout: int = 30
    retry_attempts: int = 3

    @classmethod
    def from_environment(cls) -> DatabaseProviderConfig:
        """Load database configuration from environment variables.

        Environment variables:
            DB_TYPE: Database type (sqlite, postgresql)
            DB_CONNECTION_STRING: Full connection string
            DB_MIN_POOL_SIZE: Minimum connection pool size (default: 5)
            DB_MAX_POOL_SIZE: Maximum connection pool size (default: 20)
            DB_COMMAND_TIMEOUT: Command timeout in seconds (default: 60)
            DB_CONNECTION_TIMEOUT: Connection timeout in seconds (default: 30)
            DB_RETRY_ATTEMPTS: Number of retry attempts (default: 3)

        Returns:
            Configured database provider instance.
        """
        db_type_str = ConfigLoader.get_env_var("DB_TYPE")
        database_type = DatabaseType(db_type_str.lower())

        return cls(
            database_type=database_type,
            connection_string=ConfigLoader.get_env_var("DB_CONNECTION_STRING"),
            min_pool_size=ConfigLoader.get_env_int(
                "DB_MIN_POOL_SIZE", default=5, required=False
            ),
            max_pool_size=ConfigLoader.get_env_int(
                "DB_MAX_POOL_SIZE", default=20, required=False
            ),
            command_timeout=ConfigLoader.get_env_int(
                "DB_COMMAND_TIMEOUT", default=60, required=False
            ),
            connection_timeout=ConfigLoader.get_env_int(
                "DB_CONNECTION_TIMEOUT", default=30, required=False
            ),
            retry_attempts=ConfigLoader.get_env_int(
                "DB_RETRY_ATTEMPTS", default=3, required=False
            ),
        )

    @classmethod
    def from_config_data(cls, data: DatabaseConfigData) -> DatabaseProviderConfig:
        """Load configuration from dictionary.

        Args:
            data: Configuration data.

        Returns:
            Configured database provider instance.
        """
        database_type = DatabaseType(data.database_type)

        return cls(
            database_type=database_type,
            connection_string=data.connection_string,
            min_pool_size=data.min_pool_size,
            max_pool_size=data.max_pool_size,
            command_timeout=data.command_timeout,
            connection_timeout=data.connection_timeout,
            retry_attempts=data.retry_attempts,
        )

    def validate(self) -> list[str]:
        """Validate database configuration.

        Returns:
            List of validation errors.
        """
        errors = []

        # Validate database type
        errors.extend(
            ConfigValidator.validate_enum_value(
                self.database_type.value, DatabaseType, "Database type"
            )
        )

        # Validate connection string
        errors.extend(
            ConfigValidator.validate_required_string(
                self.connection_string, "Connection string"
            )
        )

        # Validate pool sizes
        if self.min_pool_size <= 0:
            errors.append("Minimum pool size must be greater than 0")
        if self.max_pool_size <= 0:
            errors.append("Maximum pool size must be greater than 0")
        if self.min_pool_size > self.max_pool_size:
            errors.append("Minimum pool size cannot exceed maximum pool size")

        # Validate timeouts
        errors.extend(
            ConfigValidator.validate_timeout(self.command_timeout, "Command timeout")
        )
        errors.extend(
            ConfigValidator.validate_timeout(
                self.connection_timeout, "Connection timeout"
            )
        )

        # Validate retry attempts
        if self.retry_attempts < 0 or self.retry_attempts > 10:
            errors.append("Retry attempts must be between 0 and 10")

        return errors

    def to_config_data(self) -> DatabaseConfigData:
        """Convert configuration to typed data.

        Returns:
            Typed data representation.
        """
        return DatabaseConfigData(
            database_type=self.database_type.value,
            connection_string=self.connection_string,
            min_pool_size=self.min_pool_size,
            max_pool_size=self.max_pool_size,
            command_timeout=self.command_timeout,
            connection_timeout=self.connection_timeout,
            retry_attempts=self.retry_attempts,
        )

    def is_sqlite(self) -> bool:
        """Check if this is SQLite configuration.

        Returns:
            True if database type is SQLite.
        """
        return self.database_type == DatabaseType.SQLITE

    def is_postgresql(self) -> bool:
        """Check if this is PostgreSQL configuration.

        Returns:
            True if database type is PostgreSQL.
        """
        return self.database_type == DatabaseType.POSTGRESQL

    def get_safe_connection_string(self) -> str:
        """Get connection string with sensitive data masked.

        Returns:
            Safe connection string with passwords masked.
        """
        if "password=" in self.connection_string.lower():
            # Simple password masking for common formats
            return re.sub(
                r"password=[^;]+",
                "password=***",
                self.connection_string,
                flags=re.IGNORECASE,
            )
        return self.connection_string

    @classmethod
    def create_sqlite_config(cls, database_path: str) -> DatabaseProviderConfig:
        """Create configuration for SQLite database.

        Args:
            database_path: Path to SQLite database file.

        Returns:
            Configured SQLite instance.
        """
        return cls(
            database_type=DatabaseType.SQLITE,
            connection_string=f"sqlite:///{database_path}",
            min_pool_size=1,  # SQLite doesn't need large pools
            max_pool_size=5,
            command_timeout=30,
            connection_timeout=10,
            retry_attempts=3,
        )

    @classmethod
    def create_postgresql_config(
        cls,
        host: str,
        port: int,
        database: str,
        username: str,
        password: str,
    ) -> DatabaseProviderConfig:
        """Create configuration for PostgreSQL database.

        Args:
            host: PostgreSQL host.
            port: PostgreSQL port.
            database: Database name.
            username: Username.
            password: Password.

        Returns:
            Configured PostgreSQL instance.
        """
        connection_string = (
            f"postgresql://{username}:{password}@{host}:{port}/{database}"
        )

        return cls(
            database_type=DatabaseType.POSTGRESQL,
            connection_string=connection_string,
            min_pool_size=5,
            max_pool_size=20,
            command_timeout=60,
            connection_timeout=30,
            retry_attempts=3,
        )
