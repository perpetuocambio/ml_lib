"""Event Bus implementation.

Central dispatcher for domain events with async processing.
"""

import logging
import asyncio
from typing import Type
from collections import defaultdict
from ml_lib.diffusion.domain.events.base import (
    IDomainEvent,
    IEventHandler,
    IEventBus,
    EventMetadata,
)

logger = logging.getLogger(__name__)


class EventBus(IEventBus):
    """
    Simple event bus implementation with async support.

    Features:
    - Multiple handlers per event type
    - Async event processing
    - Error isolation (handler failures don't affect others)
    - Event processing metrics
    - Handler execution logging

    Unlike CommandBus/QueryBus:
    - Events can have 0 to N handlers (not exactly 1)
    - Handlers are called asynchronously
    - Handler failures are logged but don't stop execution
    - No return values expected

    Example:
        bus = EventBus()

        # Subscribe handlers
        bus.subscribe(ImageGeneratedEvent, LoggingHandler())
        bus.subscribe(ImageGeneratedEvent, NotificationHandler())
        bus.subscribe(ImageGeneratedEvent, MetricsHandler())

        # Publish event (all 3 handlers will be called)
        await bus.publish(ImageGeneratedEvent.create(
            image_path="/output/image.png",
            prompt="anime girl"
        ))
    """

    def __init__(self, enable_metrics: bool = True):
        """
        Initialize event bus.

        Args:
            enable_metrics: Enable processing time metrics
        """
        # Use defaultdict to support multiple handlers per event
        self._handlers: dict[Type[IDomainEvent], list[IEventHandler]] = defaultdict(list)
        self._enable_metrics = enable_metrics
        logger.info("EventBus initialized")

    def subscribe(
        self,
        event_type: Type[IDomainEvent],
        handler: IEventHandler
    ) -> None:
        """
        Subscribe handler to event type.

        Multiple handlers can subscribe to same event.

        Args:
            event_type: Event class
            handler: Handler instance
        """
        self._handlers[event_type].append(handler)
        handler_name = handler.__class__.__name__
        event_name = event_type.__name__
        logger.debug(f"Subscribed {handler_name} to {event_name}")

    async def publish(self, event: IDomainEvent) -> EventMetadata:
        """
        Publish event to all subscribed handlers.

        All handlers are called asynchronously in parallel.
        Handler failures are logged but don't affect other handlers.

        Args:
            event: Event to publish

        Returns:
            EventMetadata with processing information
        """
        event_type = type(event)
        event_name = event_type.__name__
        handlers = self._handlers.get(event_type, [])

        if not handlers:
            logger.debug(f"No handlers registered for {event_name}")
            return EventMetadata(
                event_type=event_name,
                handler_count=0,
                processing_time_ms=0.0,
                failed_handlers=[],
                success=True,
            )

        logger.debug(f"Publishing {event_name} to {len(handlers)} handler(s)")

        # Track metrics
        if self._enable_metrics:
            import time
            start_time = time.time()

        # Execute all handlers in parallel
        failed_handlers: list[str] = []
        tasks = []

        for handler in handlers:
            task = self._execute_handler(handler, event, failed_handlers)
            tasks.append(task)

        # Wait for all handlers to complete
        await asyncio.gather(*tasks, return_exceptions=True)

        # Calculate metrics
        if self._enable_metrics:
            elapsed_ms = (time.time() - start_time) * 1000
        else:
            elapsed_ms = 0.0

        metadata = EventMetadata(
            event_type=event_name,
            handler_count=len(handlers),
            processing_time_ms=elapsed_ms,
            failed_handlers=failed_handlers,
            success=len(failed_handlers) == 0,
        )

        if metadata.has_failures:
            logger.warning(
                f"{event_name} processing completed with {len(failed_handlers)} failure(s)"
            )
        else:
            logger.debug(
                f"{event_name} processed by {len(handlers)} handler(s) in {elapsed_ms:.2f}ms"
            )

        return metadata

    async def _execute_handler(
        self,
        handler: IEventHandler,
        event: IDomainEvent,
        failed_handlers: list[str]
    ) -> None:
        """
        Execute single handler with error isolation.

        Args:
            handler: Handler to execute
            event: Event to pass to handler
            failed_handlers: List to append to if handler fails
        """
        handler_name = handler.__class__.__name__
        event_name = type(event).__name__

        try:
            await handler.handle(event)
            logger.debug(f"{handler_name} handled {event_name}")
        except Exception as e:
            # Log error but don't propagate (error isolation)
            logger.exception(
                f"Exception in {handler_name} while handling {event_name}: {str(e)}"
            )
            failed_handlers.append(handler_name)

    def unsubscribe(
        self,
        event_type: Type[IDomainEvent],
        handler: IEventHandler
    ) -> None:
        """
        Unsubscribe handler from event type.

        Args:
            event_type: Event class
            handler: Handler instance to remove
        """
        if event_type in self._handlers:
            try:
                self._handlers[event_type].remove(handler)
                handler_name = handler.__class__.__name__
                event_name = event_type.__name__
                logger.debug(f"Unsubscribed {handler_name} from {event_name}")
            except ValueError:
                # Handler wasn't subscribed, ignore
                pass

    def get_handler_count(self, event_type: Type[IDomainEvent]) -> int:
        """
        Get number of handlers subscribed to event type.

        Args:
            event_type: Event class

        Returns:
            Number of subscribed handlers
        """
        return len(self._handlers.get(event_type, []))

    def get_all_event_types(self) -> list[Type[IDomainEvent]]:
        """
        Get list of all event types with subscribed handlers.

        Returns:
            List of event types
        """
        return list(self._handlers.keys())

    def clear(self) -> None:
        """
        Clear all handler subscriptions.

        Useful for testing.
        """
        self._handlers.clear()
        logger.debug("EventBus cleared")
