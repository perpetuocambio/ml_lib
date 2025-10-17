"""Query Bus implementation.

Central dispatcher for queries with caching and monitoring support.
"""

import logging
from typing import Type
from ml_lib.diffusion.application.queries.base import (
    IQuery,
    IQueryHandler,
    IQueryBus,
    QueryResult,
)

logger = logging.getLogger(__name__)


class QueryBus(IQueryBus):
    """
    Simple query bus implementation.

    Features:
    - Handler registration by query type
    - Query dispatching to appropriate handler
    - Error handling and logging
    - Performance monitoring
    - Cache-ready (future enhancement)

    Unlike CommandBus:
    - No complex status handling (queries succeed or raise)
    - Focus on read performance
    - Cacheable results

    Example:
        bus = QueryBus()
        bus.register(GetLoRAQuery, GetLoRAHandler(repository))
        result = bus.dispatch(GetLoRAQuery(lora_id="123"))
    """

    def __init__(self, enable_monitoring: bool = True):
        """
        Initialize query bus.

        Args:
            enable_monitoring: Enable performance monitoring
        """
        self._handlers: dict[Type[IQuery], IQueryHandler] = {}
        self._enable_monitoring = enable_monitoring
        logger.info("QueryBus initialized")

    def register(self, query_type: Type[IQuery], handler: IQueryHandler) -> None:
        """
        Register handler for query type.

        Args:
            query_type: Query class
            handler: Handler instance

        Raises:
            ValueError: If handler already registered for query type
        """
        if query_type in self._handlers:
            raise ValueError(
                f"Handler already registered for {query_type.__name__}"
            )

        self._handlers[query_type] = handler
        logger.debug(f"Registered handler for {query_type.__name__}")

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
        query_type = type(query)
        query_name = query_type.__name__

        # Find handler
        handler = self._handlers.get(query_type)
        if handler is None:
            error_msg = f"No handler registered for {query_name}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Execute handler with monitoring
        try:
            logger.debug(f"Dispatching query: {query_name}")

            if self._enable_monitoring:
                import time
                start_time = time.time()
                result = handler.handle(query)
                elapsed_ms = (time.time() - start_time) * 1000

                # Add monitoring metadata
                if result.metadata is None:
                    metadata = {}
                else:
                    metadata = dict(result.metadata)

                metadata["query_time_ms"] = elapsed_ms
                metadata["query_name"] = query_name

                result = QueryResult(data=result.data, metadata=metadata)

                logger.debug(f"{query_name} executed in {elapsed_ms:.2f}ms")
            else:
                result = handler.handle(query)

            return result

        except Exception as e:
            logger.exception(f"Exception in {query_name} handler: {str(e)}")
            raise

    def is_registered(self, query_type: Type[IQuery]) -> bool:
        """
        Check if handler is registered for query type.

        Args:
            query_type: Query class

        Returns:
            True if handler registered, False otherwise
        """
        return query_type in self._handlers

    def unregister(self, query_type: Type[IQuery]) -> None:
        """
        Unregister handler for query type.

        Args:
            query_type: Query class
        """
        if query_type in self._handlers:
            del self._handlers[query_type]
            logger.debug(f"Unregistered handler for {query_type.__name__}")

    def get_registered_queries(self) -> list[Type[IQuery]]:
        """
        Get list of registered query types.

        Returns:
            List of query types with registered handlers
        """
        return list(self._handlers.keys())
