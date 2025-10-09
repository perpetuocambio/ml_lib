"""Database type enumeration."""

from enum import Enum


class DatabaseType(Enum):
    """Supported database types."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
