"""Domain interface for database connections."""

from abc import ABC, abstractmethod

from infrastructure.persistence.database.query_parameters import QueryParameters
from infrastructure.persistence.database.query_result import QueryResult


class IDatabaseConnection(ABC):
    """Domain interface for database connection operations."""

    @abstractmethod
    async def execute_query(
        self, query: str, params: QueryParameters | None = None
    ) -> QueryResult:
        """Execute a query and return results."""
        pass

    @abstractmethod
    async def execute_single(
        self, query: str, params: QueryParameters | None = None
    ) -> QueryResult:
        """Execute a query expecting single result."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass
