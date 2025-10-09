"""Database configuration entity."""

from dataclasses import dataclass

from infrastructure.persistence.database.enums.database_type import DatabaseType


@dataclass
class DatabaseConfig:
    """Database configuration settings."""

    database_type: DatabaseType
    connection_string: str
    min_pool_size: int = 5
    max_pool_size: int = 20
    command_timeout: int = 60
