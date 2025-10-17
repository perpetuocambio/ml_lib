"""Base Query interfaces and types.

Defines the core Query Pattern interfaces following CQRS principles.
Queries are read-only operations with no side effects.
"""

from typing import Protocol, TypeVar, Generic, runtime_checkable, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryResult:
    """Result of query execution.

    Queries always succeed or raise exceptions.
    Unlike commands, there's no complex status handling needed.

    Attributes:
        data: Query result data
        metadata: Optional metadata (e.g., cache hit, query time)
    """

    data: Any
    metadata: dict[str, Any] | None = None

    @classmethod
    def success(cls, data: Any, metadata: dict | None = None) -> "QueryResult":
        """Create successful query result."""
        return cls(data=data, metadata=metadata)


@runtime_checkable
class IQuery(Protocol):
    """
    Base interface for all queries.

    Queries are immutable data structures that represent read requests.
    They should be frozen dataclasses with all required data for execution.

    Unlike Commands:
    - Queries don't modify state
    - Queries return data directly
    - Queries can be cached
    - Queries can use optimized read models

    Example:
        @dataclass(frozen=True)
        class GetLoRAByIdQuery(IQuery):
            lora_id: str
    """

    pass


# Type variable for query generic typing
TQuery = TypeVar("TQuery", bound=IQuery, contravariant=True)


@runtime_checkable
class IQueryHandler(Protocol, Generic[TQuery]):
    """
    Handler for a specific query type.

    Each query should have exactly one handler (Single Responsibility).
    Handlers contain the logic for fetching and projecting data.

    Unlike Command Handlers:
    - No validation needed (read-only)
    - No status codes (success or exception)
    - Focus on performance and caching
    - Can use denormalized read models

    Example:
        class GetLoRAByIdHandler(IQueryHandler[GetLoRAByIdQuery]):
            def __init__(self, repository: ILoRARepository):
                self.repository = repository

            def handle(self, query: GetLoRAByIdQuery) -> QueryResult:
                lora = self.repository.get_by_id(query.lora_id)
                return QueryResult.success(lora)
    """

    def handle(self, query: TQuery) -> QueryResult:
        """
        Handle query execution.

        Args:
            query: Query to execute

        Returns:
            QueryResult with data

        Raises:
            Exception: If query fails (not found, database error, etc.)
        """
        ...


@runtime_checkable
class IQueryBus(Protocol):
    """
    Query bus for dispatching queries to handlers.

    Similar to CommandBus but optimized for reads:
    - Caching support
    - Read-only guarantees
    - Performance monitoring

    Example:
        bus = QueryBus()
        bus.register(GetLoRAByIdQuery, GetLoRAByIdHandler(repository))
        result = bus.dispatch(GetLoRAByIdQuery(lora_id="123"))
        lora = result.data
    """

    def register(self, query_type: type[IQuery], handler: IQueryHandler) -> None:
        """
        Register handler for query type.

        Args:
            query_type: Query class
            handler: Handler instance
        """
        ...

    def dispatch(self, query: IQuery) -> QueryResult:
        """
        Dispatch query to appropriate handler.

        Args:
            query: Query to execute

        Returns:
            QueryResult with data

        Raises:
            ValueError: If no handler registered for query type
            Exception: If query execution fails
        """
        ...
