"""Domain Events - Observer Pattern implementation.

Events represent things that have happened in the domain.
They enable loose coupling and reactive architectures.

Observer Pattern benefits:
- Decoupled communication between components
- Multiple observers can react to same event
- Easy to add new event handlers without modifying existing code
- Async processing support
- Side effects are isolated from core domain logic

Event Flow:
    Domain Service → Event.create() → EventBus.publish() → Handler.handle()

Example:
    # Setup event bus
    bus = EventBus()
    bus.subscribe(ImageGeneratedEvent, LoggingHandler())
    bus.subscribe(ImageGeneratedEvent, MetricsHandler())
    bus.subscribe(ImageGeneratedEvent, NotificationHandler())

    # Domain service publishes event
    event = ImageGeneratedEvent.create(
        image_path="/output/image.png",
        prompt="anime girl",
        base_model="SDXL",
        loras_used=["style-v1"],
        generation_time_seconds=12.5,
        seed=42,
    )
    await bus.publish(event)  # All 3 handlers will be called

Design Patterns:
- Observer Pattern: Event handlers observe domain events
- Publish-Subscribe: EventBus mediates between publishers and subscribers
- Event Sourcing ready: Events are immutable facts
"""

from ml_lib.diffusion.domain.events.base import (
    IDomainEvent,
    IEventHandler,
    IEventBus,
    EventMetadata,
)
from ml_lib.diffusion.domain.events.bus import EventBus
from ml_lib.diffusion.domain.events.lora_events import (
    LoRAsRecommendedEvent,
    TopLoRARecommendedEvent,
    LoRALoadedEvent,
    LoRAFilteredEvent,
)
from ml_lib.diffusion.domain.events.image_events import (
    ImageGenerationRequestedEvent,
    ImageGeneratedEvent,
    ImageGenerationFailedEvent,
    PromptAnalyzedEvent,
)
from ml_lib.diffusion.domain.events.handlers import (
    LoggingEventHandler,
    MetricsEventHandler,
    ErrorLoggingHandler,
    CachingHandler,
    PromptAnalyticsHandler,
    MultiEventHandler,
)

__all__ = [
    # Base interfaces
    "IDomainEvent",
    "IEventHandler",
    "IEventBus",
    "EventMetadata",
    # Event bus
    "EventBus",
    # LoRA events
    "LoRAsRecommendedEvent",
    "TopLoRARecommendedEvent",
    "LoRALoadedEvent",
    "LoRAFilteredEvent",
    # Image events
    "ImageGenerationRequestedEvent",
    "ImageGeneratedEvent",
    "ImageGenerationFailedEvent",
    "PromptAnalyzedEvent",
    # Example handlers
    "LoggingEventHandler",
    "MetricsEventHandler",
    "ErrorLoggingHandler",
    "CachingHandler",
    "PromptAnalyticsHandler",
    "MultiEventHandler",
]
