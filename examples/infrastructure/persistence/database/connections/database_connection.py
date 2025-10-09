"""Abstract database connection interface."""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from infrastructure.persistence.database.entities.database_health_check_result import (
    DatabaseHealthCheckResult,
)


class IDatabaseConnection(ABC):
    """Abstract interface for database connections."""

    @abstractmethod
    async def initialize_pool(self) -> None:
        """Initialize connection pool."""
        pass

    @abstractmethod
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator:
        """Get a database connection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the connection pool."""
        pass

    @abstractmethod
    async def health_check(self) -> DatabaseHealthCheckResult:
        """Perform database health check."""
        pass

    @abstractmethod
    async def setup_vector_tables(self) -> None:
        """Setup vector tables for RAG operations."""
        pass

    @abstractmethod
    async def initialize_schema(self) -> None:
        """Initialize complete database schema."""
        pass
