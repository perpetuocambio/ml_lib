# Observer Pattern & Domain Events Guide

## Overview

**Observer Pattern** enables loose coupling between objects through event-driven communication:

- **Events**: Immutable records of things that happened (past tense)
- **Publishers**: Domain services that emit events
- **Subscribers**: Event handlers that react to events
- **Event Bus**: Mediator that dispatches events to handlers

## Architecture

```
┌─────────────────────────────────────────────┐
│          Domain Service Layer               │
│                                             │
│  Service executes business logic           │
│         ↓                                   │
│  Creates domain event (past tense)          │
│         ↓                                   │
│  Publishes event to EventBus                │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│             EventBus (Mediator)             │
│                                             │
│  - Receives event from publisher            │
│  - Finds all subscribed handlers            │
│  - Dispatches event to handlers (async)     │
│  - Isolates handler failures                │
└─────────────────┬───────────────────────────┘
                  │
                  ├──────────┬──────────┬──────────┐
                  ▼          ▼          ▼          ▼
           ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
           │ Handler  │ │ Handler  │ │ Handler  │ │ Handler  │
           │    1     │ │    2     │ │    3     │ │    N     │
           └──────────┘ └──────────┘ └──────────┘ └──────────┘
           Logging     Metrics      Caching      Notifications
```

## Key Concepts

### 1. Domain Events

**Definition:** Immutable facts about things that happened in the domain.

**Characteristics:**
- Named in past tense (`ImageGeneratedEvent`, not `GenerateImageEvent`)
- Frozen dataclasses (immutable)
- Contain all data needed by handlers
- Include metadata (event_id, occurred_at)

**Example:**
```python
@dataclass(frozen=True)
class ImageGeneratedEvent(IDomainEvent):
    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=datetime.now)

    # Domain data
    image_path: str
    prompt: str
    base_model: str
    loras_used: list[str]
    generation_time_seconds: float
    seed: int
```

### 2. Event Handlers

**Definition:** Observers that react to domain events.

**Characteristics:**
- Implement `IEventHandler[TEvent]`
- Async methods (`async def handle(...)`)
- Should be idempotent (safe to run multiple times)
- Should not raise exceptions (log errors instead)
- Multiple handlers can observe same event

**Example:**
```python
class MetricsEventHandler(IEventHandler[ImageGeneratedEvent]):
    async def handle(self, event: ImageGeneratedEvent) -> None:
        # Track metrics
        self.total_images += 1
        self.total_time += event.generation_time_seconds

        # Log to monitoring system
        logger.info(f"Image generated in {event.generation_time_seconds:.2f}s")
```

### 3. Event Bus

**Definition:** Mediator that dispatches events to handlers.

**Characteristics:**
- Async event processing
- Multiple handlers per event type (0 to N)
- Error isolation (one handler failure doesn't affect others)
- Performance monitoring

**Example:**
```python
# Setup
bus = EventBus()
bus.subscribe(ImageGeneratedEvent, LoggingHandler())
bus.subscribe(ImageGeneratedEvent, MetricsHandler())
bus.subscribe(ImageGeneratedEvent, NotificationHandler())

# Publish
event = ImageGeneratedEvent.create(
    image_path="/output/image.png",
    prompt="anime girl",
    base_model="SDXL",
    loras_used=["style-v1"],
    generation_time_seconds=12.5,
    seed=42,
)
await bus.publish(event)  # All 3 handlers called
```

## Available Events

### LoRA Events

#### LoRAsRecommendedEvent
Emitted when LoRAs are recommended for a prompt.

```python
event = LoRAsRecommendedEvent.create(
    prompt="anime girl with blue hair",
    base_model="SDXL",
    lora_ids=["lora-123", "lora-456"],
    recommendation_count=3,
    confidence_threshold=0.7,
)
```

**Use cases:**
- Recommendation history logging
- Analytics on recommendation patterns
- Caching popular recommendations

#### TopLoRARecommendedEvent
Emitted when single best LoRA is recommended.

```python
event = TopLoRARecommendedEvent.create(
    prompt="cyberpunk city",
    base_model="SDXL",
    lora_id="cyberpunk-lora-v2",
    confidence=0.95,
)
```

#### LoRALoadedEvent
Emitted when a LoRA model is loaded from repository.

```python
event = LoRALoadedEvent.create(
    lora_id="anime-style-v1",
    lora_name="Anime Art Style V1",
    base_model="SDXL",
)
```

**Use cases:**
- Cache warming for frequently loaded LoRAs
- Performance monitoring
- Repository access patterns

#### LoRAFilteredEvent
Emitted when LoRAs are filtered by confidence threshold.

```python
event = LoRAFilteredEvent.create(
    original_count=10,
    filtered_count=3,
    confidence_threshold=0.7,
    base_model="SDXL",
)
```

### Image Generation Events

#### ImageGenerationRequestedEvent
Emitted when image generation is requested.

```python
event = ImageGenerationRequestedEvent.create(
    prompt="anime girl with blue hair",
    negative_prompt="bad quality",
    base_model="SDXL",
    loras_requested=["style-lora-v1"],
    seed=42,
)
```

#### ImageGeneratedEvent
Emitted when image generation completes successfully.

```python
event = ImageGeneratedEvent.create(
    image_path="/output/image_123.png",
    prompt="anime girl",
    base_model="SDXL",
    loras_used=["style-lora-v1"],
    generation_time_seconds=12.5,
    seed=42,
)
```

#### ImageGenerationFailedEvent
Emitted when image generation fails.

```python
event = ImageGenerationFailedEvent.create(
    prompt="anime girl",
    base_model="SDXL",
    error_message="Out of memory",
    error_type="RuntimeError",
    loras_attempted=["heavy-lora-v1"],
)
```

#### PromptAnalyzedEvent
Emitted when a prompt is analyzed.

```python
event = PromptAnalyzedEvent.create(
    original_prompt="girl with blue hair",
    optimized_prompt="anime girl, blue hair, detailed",
    detected_intent="character_portrait",
    complexity_score=0.6,
    token_count=45,
)
```

## Example Event Handlers

### LoggingEventHandler
Logs LoRA recommendations for audit trail.

```python
class LoggingEventHandler(IEventHandler[LoRAsRecommendedEvent]):
    async def handle(self, event: LoRAsRecommendedEvent) -> None:
        logger.info(
            f"LoRAs recommended: {event.recommendation_count} LoRAs "
            f"for prompt '{event.prompt}'"
        )
```

### MetricsEventHandler
Tracks image generation performance metrics.

```python
class MetricsEventHandler(IEventHandler[ImageGeneratedEvent]):
    def __init__(self):
        self.total_images_generated = 0
        self.total_generation_time = 0.0

    async def handle(self, event: ImageGeneratedEvent) -> None:
        self.total_images_generated += 1
        self.total_generation_time += event.generation_time_seconds
        avg_time = self.total_generation_time / self.total_images_generated
        logger.info(f"Avg generation time: {avg_time:.2f}s")
```

### ErrorLoggingHandler
Logs and tracks image generation failures.

```python
class ErrorLoggingHandler(IEventHandler[ImageGenerationFailedEvent]):
    def __init__(self):
        self.error_count = 0

    async def handle(self, event: ImageGenerationFailedEvent) -> None:
        self.error_count += 1
        logger.error(
            f"Generation failed: {event.error_message} "
            f"(total failures: {self.error_count})"
        )
```

### CachingHandler
Tracks frequently loaded LoRAs for cache optimization.

```python
class CachingHandler(IEventHandler[LoRALoadedEvent]):
    def __init__(self):
        self.load_counts: dict[str, int] = {}

    async def handle(self, event: LoRALoadedEvent) -> None:
        lora_id = event.lora_id
        self.load_counts[lora_id] = self.load_counts.get(lora_id, 0) + 1

        if self.load_counts[lora_id] >= 5:
            logger.info(f"LoRA '{event.lora_name}' loaded {self.load_counts[lora_id]} times - consider caching")
```

## Usage Patterns

### Pattern 1: Single Event, Multiple Handlers

Multiple handlers react to same event for different purposes.

```python
# Setup
bus = EventBus()
bus.subscribe(ImageGeneratedEvent, LoggingHandler())
bus.subscribe(ImageGeneratedEvent, MetricsHandler())
bus.subscribe(ImageGeneratedEvent, NotificationHandler())
bus.subscribe(ImageGeneratedEvent, CatalogingHandler())

# One event, four reactions
event = ImageGeneratedEvent.create(...)
await bus.publish(event)
```

### Pattern 2: Handler Subscribing to Multiple Events

One handler reacts to multiple event types.

```python
class UnifiedMonitoringHandler(
    IEventHandler[ImageGeneratedEvent],
    IEventHandler[ImageGenerationFailedEvent],
):
    async def handle(
        self,
        event: ImageGeneratedEvent | ImageGenerationFailedEvent
    ) -> None:
        if isinstance(event, ImageGeneratedEvent):
            self.track_success(event)
        elif isinstance(event, ImageGenerationFailedEvent):
            self.track_failure(event)

# Subscribe to both events
bus.subscribe(ImageGeneratedEvent, handler)
bus.subscribe(ImageGenerationFailedEvent, handler)
```

### Pattern 3: Domain Service Publishing Events

Domain services publish events after state changes.

```python
class LoRARecommendationService:
    def __init__(self, repository, event_bus):
        self.repository = repository
        self.event_bus = event_bus

    async def recommend_loras(self, prompt, base_model):
        # Execute business logic
        recommendations = self._get_recommendations(prompt, base_model)

        # Publish event (fire and forget)
        event = LoRAsRecommendedEvent.create(
            prompt=prompt,
            base_model=base_model,
            lora_ids=[rec.lora.id for rec in recommendations],
            recommendation_count=len(recommendations),
            confidence_threshold=0.0,
        )
        await self.event_bus.publish(event)

        return recommendations
```

### Pattern 4: Event-Driven Workflows

Chain events to create workflows.

```python
class ImageGenerationWorkflow:
    """
    Example workflow:
    1. Request received → ImageGenerationRequestedEvent
    2. Prompt analyzed → PromptAnalyzedEvent
    3. LoRAs recommended → LoRAsRecommendedEvent
    4. Image generated → ImageGeneratedEvent
    """

    def __init__(self, event_bus):
        self.event_bus = event_bus

        # Handler triggers next step
        bus.subscribe(ImageGenerationRequestedEvent, self.analyze_prompt_handler)
        bus.subscribe(PromptAnalyzedEvent, self.recommend_loras_handler)
        bus.subscribe(LoRAsRecommendedEvent, self.generate_image_handler)
```

## Benefits of Observer Pattern

### 1. Loose Coupling
Domain services don't know about event handlers.
Handlers can be added/removed without modifying services.

```python
# Service publishes event, doesn't know who's listening
await event_bus.publish(event)

# New handler can be added anytime
bus.subscribe(ImageGeneratedEvent, NewFeatureHandler())
```

### 2. Open/Closed Principle (SOLID-O)
New functionality via new handlers, not modifying existing code.

```python
# Add new feature without touching existing code
class EmailNotificationHandler(IEventHandler[ImageGeneratedEvent]):
    async def handle(self, event: ImageGeneratedEvent) -> None:
        await send_email(f"Your image is ready: {event.image_path}")

bus.subscribe(ImageGeneratedEvent, EmailNotificationHandler())
```

### 3. Async Processing
Handlers run asynchronously, improving responsiveness.

```python
# All handlers run in parallel
await bus.publish(event)  # Returns immediately after dispatching
```

### 4. Error Isolation
One handler failure doesn't affect others.

```python
# Even if MetricsHandler fails, LoggingHandler still runs
bus.subscribe(event_type, LoggingHandler())
bus.subscribe(event_type, MetricsHandler())  # May fail
bus.subscribe(event_type, NotificationHandler())  # Still runs
```

### 5. Testability
Easy to test handlers in isolation.

```python
async def test_metrics_handler():
    handler = MetricsEventHandler()
    event = ImageGeneratedEvent.create(
        image_path="/test.png",
        prompt="test",
        base_model="SDXL",
        loras_used=[],
        generation_time_seconds=10.0,
        seed=42,
    )

    await handler.handle(event)

    assert handler.total_images_generated == 1
    assert handler.get_average_generation_time() == 10.0
```

## Best Practices

### 1. Events are Immutable Facts

```python
# ✅ Good: Past tense, immutable
@dataclass(frozen=True)
class ImageGeneratedEvent(IDomainEvent):
    image_path: str
    prompt: str

# ❌ Bad: Present/future tense, mutable
@dataclass
class GenerateImageEvent:  # Wrong tense
    image_path: str  # Mutable
```

### 2. Events Contain All Needed Data

```python
# ✅ Good: Complete data
event = ImageGeneratedEvent.create(
    image_path="/output/image.png",
    prompt="anime girl",
    base_model="SDXL",
    loras_used=["style-v1"],
    generation_time_seconds=12.5,
    seed=42,
)

# ❌ Bad: Incomplete (handlers need to fetch data)
event = ImageGeneratedEvent.create(
    image_path="/output/image.png"  # Missing context
)
```

### 3. Handlers are Idempotent

```python
# ✅ Good: Can run multiple times safely
class MetricsHandler(IEventHandler[ImageGeneratedEvent]):
    async def handle(self, event: ImageGeneratedEvent) -> None:
        metrics.increment(f"images.{event.base_model}")

# ❌ Bad: Not idempotent
class BadHandler(IEventHandler[ImageGeneratedEvent]):
    async def handle(self, event: ImageGeneratedEvent) -> None:
        self.counter += 1  # Running twice gives wrong result
```

### 4. Handlers Don't Raise Exceptions

```python
# ✅ Good: Log errors, don't raise
class SafeHandler(IEventHandler[ImageGeneratedEvent]):
    async def handle(self, event: ImageGeneratedEvent) -> None:
        try:
            await self.external_service.notify(event)
        except Exception as e:
            logger.error(f"Notification failed: {e}")
            # Don't raise, allow other handlers to run

# ❌ Bad: Raises exceptions
class BadHandler(IEventHandler[ImageGeneratedEvent]):
    async def handle(self, event: ImageGeneratedEvent) -> None:
        await self.external_service.notify(event)  # May raise
```

### 5. Publish Events After State Changes

```python
# ✅ Good: Publish after successful state change
async def generate_image(request):
    # Execute business logic
    result = await self.generator.generate(request)

    # State changed successfully, publish event
    event = ImageGeneratedEvent.create(...)
    await self.event_bus.publish(event)

    return result

# ❌ Bad: Publish before state change
async def generate_image(request):
    event = ImageGeneratedEvent.create(...)
    await self.event_bus.publish(event)  # Too early!

    result = await self.generator.generate(request)  # May fail
    return result
```

## Integration with CQRS

Events complement Commands and Queries:

```python
# Command modifies state
command = GenerateImageCommand(prompt="anime girl", ...)
result = command_bus.dispatch(command)

# Inside command handler:
class GenerateImageHandler:
    async def handle(self, command):
        # Execute generation
        image = await self.generator.generate(...)

        # Publish event
        event = ImageGeneratedEvent.create(...)
        await self.event_bus.publish(event)

        return CommandResult.success(image)
```

## Future Enhancements

### Event Store
Persist all events for audit trail and event sourcing.

```python
class EventStore:
    async def append(self, event: IDomainEvent) -> None:
        await self.db.insert(event)

    async def get_events(self, aggregate_id: str) -> list[IDomainEvent]:
        return await self.db.query(aggregate_id)
```

### Event Replay
Replay events to rebuild state or debug issues.

```python
async def replay_events(events: list[IDomainEvent]):
    for event in events:
        await event_bus.publish(event)
```

### Distributed Events
Publish events to message broker (RabbitMQ, Kafka).

```python
class DistributedEventBus(IEventBus):
    async def publish(self, event: IDomainEvent) -> None:
        await self.kafka.send(topic="domain-events", value=event)
```

## Summary

Observer Pattern with Domain Events provides:
- ✅ Loose coupling between components
- ✅ Open/Closed principle compliance
- ✅ Async, scalable architecture
- ✅ Easy to add new features
- ✅ Excellent testability
- ✅ Event-driven workflows

Use **Events** for notifications about things that happened.
Use **Commands** for requests to change state.
Use **Queries** for requests to read data.
