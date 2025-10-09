"""Database configuration factory."""

import os
from pathlib import Path

from infrastructure.persistence.database.config.database_config_entity import (
    DatabaseConfig,
)
from infrastructure.persistence.database.enums.database_type import DatabaseType


class DatabaseConfigFactory:
    """Factory for creating database configurations."""

    @staticmethod
    def from_environment() -> DatabaseConfig:
        """Create database config from environment variables."""
        db_type_str = os.getenv("PYINTELCIVIL_DATABASE_TYPE", "sqlite").lower()

        try:
            db_type = DatabaseType(db_type_str)
        except ValueError:
            db_type = DatabaseType.SQLITE

        if db_type == DatabaseType.POSTGRESQL:
            host = os.getenv("POSTGRES_HOST", "localhost")
            port = int(os.getenv("POSTGRES_PORT", "5432"))
            database = os.getenv("POSTGRES_DATABASE", "pyintelcivil")
            user = os.getenv("POSTGRES_USER", "postgres")
            password = os.getenv("POSTGRES_PASSWORD", "postgres")

            connection_string = (
                f"postgresql://{user}:{password}@{host}:{port}/{database}"
            )

            return DatabaseConfig(
                database_type=db_type,
                connection_string=connection_string,
                min_pool_size=int(os.getenv("POSTGRES_MIN_POOL_SIZE", "5")),
                max_pool_size=int(os.getenv("POSTGRES_MAX_POOL_SIZE", "20")),
                command_timeout=int(os.getenv("POSTGRES_COMMAND_TIMEOUT", "60")),
            )
        else:
            # SQLite default
            db_path = os.getenv("SQLITE_DATABASE_PATH", "pyintelcivil.db")
            connection_string = f"sqlite:///{db_path}"

            return DatabaseConfig(
                database_type=db_type, connection_string=connection_string
            )

    @staticmethod
    def create_sqlite_config(database_path: Path) -> DatabaseConfig:
        """Create SQLite configuration."""
        return DatabaseConfig(
            database_type=DatabaseType.SQLITE,
            connection_string=f"sqlite:///{database_path}",
        )

    @staticmethod
    def create_postgresql_config(
        host: str = "localhost",
        port: int = 5432,
        database: str = "pyintelcivil",
        user: str = "postgres",
        password: str = "postgres",
    ) -> DatabaseConfig:
        """Create PostgreSQL configuration."""
        connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
        return DatabaseConfig(
            database_type=DatabaseType.POSTGRESQL, connection_string=connection_string
        )
