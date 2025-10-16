"""Base Domain Events interfaces.

Domain Events represent things that have happened in the domain.
They are immutable facts about state changes.

Following DDD (Domain-Driven Design) Event Pattern:
- Events are past-tense (ImageGenerated, LoRARecommended)
- Events are immutable (frozen dataclasses)
- Events carry all data needed by handlers
- Events enable loose coupling between components

Observer Pattern benefits:
- Decoupled communication between objects
- Multiple observers can react to same event
- Easy to add new event handlers
- Async processing support
"""

from typing import Protocol, TypeVar, Generic, runtime_checkable
from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4


@dataclass(frozen=True)
class IDomainEvent:
    """
    Base interface for all domain events.

    Domain events are immutable records of things that happened.
    They should be frozen dataclasses with all event data.

    Attributes:
        event_id: Unique identifier for this event instance
        occurred_at: When the event occurred

    Example:
        @dataclass(frozen=True)
        class ImageGeneratedEvent(IDomainEvent):
            image_path: str
            prompt: str
            loras_used: list[str]
            # Inherits event_id and occurred_at with defaults from base
    """

    # Protocol/interface - no actual implementation
    pass

    @classmethod
    def create(cls, **kwargs):
        """
        Factory method to create event with auto-generated metadata.

        Subclasses should override to provide defaults for event_id and occurred_at.

        Args:
            **kwargs: Event-specific data

        Returns:
            New event instance
        """
        return cls(
            event_id=str(uuid4()),
            occurred_at=datetime.now(),
            **kwargs
        )


# Type variable for event generic typing
TEvent = TypeVar("TEvent", bound=IDomainEvent, contravariant=True)


@runtime_checkable
class IEventHandler(Protocol, Generic[TEvent]):
    """
    Handler for a specific event type.

    Event handlers react to domain events.
    Multiple handlers can observe the same event type.

    Unlike Command/Query handlers:
    - Events can have 0 to N handlers
    - Handlers should not modify domain state directly
    - Handlers typically trigger side effects (logging, notifications, etc.)
    - Handlers can be async

    Example:
        class ImageGeneratedHandler(IEventHandler[ImageGeneratedEvent]):
            def __init__(self, notification_service: NotificationService):
                self.notifications = notification_service

            async def handle(self, event: ImageGeneratedEvent) -> None:
                await self.notifications.notify_user(
                    f"Image generated: {event.image_path}"
                )
    """

    async def handle(self, event: TEvent) -> None:
        """
        Handle domain event.

        Args:
            event: Event to handle

        Note:
            Handlers should be idempotent (safe to run multiple times).
            Handlers should not raise exceptions (log errors instead).
        """
        ...


@runtime_checkable
class IEventBus(Protocol):
    """
    Event bus for publishing events to handlers.

    Implements Observer pattern:
    - Observers (handlers) subscribe to events
    - Publishers publish events
    - Bus dispatches events to all registered handlers

    Features:
    - Multiple handlers per event type
    - Async event processing
    - Error isolation (one handler failure doesn't affect others)
    - Event history/logging

    Example:
        bus = EventBus()
        bus.subscribe(ImageGeneratedEvent, ImageGeneratedHandler(service))
        bus.subscribe(ImageGeneratedEvent, MetricsHandler(metrics))

        await bus.publish(ImageGeneratedEvent.create(
            image_path="/output/image.png",
            prompt="anime girl",
            loras_used=["style-lora-v1"]
        ))
    """

    def subscribe(
        self,
        event_type: type[IDomainEvent],
        handler: IEventHandler
    ) -> None:
        """
        Subscribe handler to event type.

        Multiple handlers can subscribe to same event type.

        Args:
            event_type: Event class
            handler: Handler instance
        """
        ...

    async def publish(self, event: IDomainEvent) -> None:
        """
        Publish event to all subscribed handlers.

        Args:
            event: Event to publish

        Note:
            All handlers are called asynchronously.
            Handler failures are logged but don't stop other handlers.
        """
        ...

    def unsubscribe(
        self,
        event_type: type[IDomainEvent],
        handler: IEventHandler
    ) -> None:
        """
        Unsubscribe handler from event type.

        Args:
            event_type: Event class
            handler: Handler instance to remove
        """
        ...


@dataclass(frozen=True)
class EventMetadata:
    """
    Metadata about event processing.

    Useful for monitoring, debugging, and audit trails.

    Attributes:
        event_type: Name of event type
        handler_count: Number of handlers that processed event
        processing_time_ms: Total time to process all handlers
        failed_handlers: List of handlers that failed
        success: Whether all handlers succeeded
    """

    event_type: str
    handler_count: int
    processing_time_ms: float
    failed_handlers: list[str]
    success: bool

    @property
    def has_failures(self) -> bool:
        """Check if any handlers failed."""
        return len(self.failed_handlers) > 0
