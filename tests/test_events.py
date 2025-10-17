"""Tests for Observer Pattern and Domain Events.

Tests for events, handlers, and EventBus.
"""

import pytest
import asyncio
from ml_lib.diffusion.domain.events import (
    EventBus,
    EventMetadata,
    # LoRA Events
    LoRAsRecommendedEvent,
    TopLoRARecommendedEvent,
    LoRALoadedEvent,
    LoRAFilteredEvent,
    # Image Events
    ImageGenerationRequestedEvent,
    ImageGeneratedEvent,
    ImageGenerationFailedEvent,
    PromptAnalyzedEvent,
    # Handlers
    LoggingEventHandler,
    MetricsEventHandler,
    ErrorLoggingHandler,
    CachingHandler,
    PromptAnalyticsHandler,
    MultiEventHandler,
)


# ============================================================================
# EventBus Tests
# ============================================================================

@pytest.mark.asyncio
async def test_event_bus_initialization():
    """Test event bus initialization."""
    bus = EventBus()
    assert bus._handlers == {}


@pytest.mark.asyncio
async def test_event_bus_subscribe():
    """Test subscribing handler to event."""
    bus = EventBus()
    handler = LoggingEventHandler()

    bus.subscribe(LoRAsRecommendedEvent, handler)

    assert bus.get_handler_count(LoRAsRecommendedEvent) == 1


@pytest.mark.asyncio
async def test_event_bus_multiple_handlers():
    """Test multiple handlers for same event."""
    bus = EventBus()
    handler1 = LoggingEventHandler()
    handler2 = LoggingEventHandler()

    bus.subscribe(LoRAsRecommendedEvent, handler1)
    bus.subscribe(LoRAsRecommendedEvent, handler2)

    assert bus.get_handler_count(LoRAsRecommendedEvent) == 2


@pytest.mark.asyncio
async def test_event_bus_unsubscribe():
    """Test unsubscribing handler."""
    bus = EventBus()
    handler = LoggingEventHandler()

    bus.subscribe(LoRAsRecommendedEvent, handler)
    assert bus.get_handler_count(LoRAsRecommendedEvent) == 1

    bus.unsubscribe(LoRAsRecommendedEvent, handler)
    assert bus.get_handler_count(LoRAsRecommendedEvent) == 0


@pytest.mark.asyncio
async def test_event_bus_publish_no_handlers():
    """Test publishing event with no handlers."""
    bus = EventBus()

    event = LoRAsRecommendedEvent.create(
        prompt="test",
        base_model="SDXL",
        lora_ids=[],
        recommendation_count=0,
    )

    metadata = await bus.publish(event)

    assert metadata.event_type == "LoRAsRecommendedEvent"
    assert metadata.handler_count == 0
    assert metadata.success is True


@pytest.mark.asyncio
async def test_event_bus_publish_with_handlers():
    """Test publishing event with handlers."""
    bus = EventBus()
    handler_called = []

    from ml_lib.diffusion.domain.events.base import IEventHandler

    class TestHandler(IEventHandler[LoRAsRecommendedEvent]):
        async def handle(self, event: LoRAsRecommendedEvent) -> None:
            handler_called.append(event)

    bus.subscribe(LoRAsRecommendedEvent, TestHandler())

    event = LoRAsRecommendedEvent.create(
        prompt="test",
        base_model="SDXL",
        lora_ids=["lora1"],
        recommendation_count=1,
    )

    metadata = await bus.publish(event)

    assert metadata.handler_count == 1
    assert metadata.success is True
    assert len(handler_called) == 1
    assert handler_called[0].prompt == "test"


@pytest.mark.asyncio
async def test_event_bus_error_isolation():
    """Test that handler errors don't affect other handlers."""
    bus = EventBus()
    successful_handler_called = []

    from ml_lib.diffusion.domain.events.base import IEventHandler

    class FailingHandler(IEventHandler[LoRAsRecommendedEvent]):
        async def handle(self, event: LoRAsRecommendedEvent) -> None:
            raise ValueError("Handler failed")

    class SuccessfulHandler(IEventHandler[LoRAsRecommendedEvent]):
        async def handle(self, event: LoRAsRecommendedEvent) -> None:
            successful_handler_called.append(event)

    bus.subscribe(LoRAsRecommendedEvent, FailingHandler())
    bus.subscribe(LoRAsRecommendedEvent, SuccessfulHandler())

    event = LoRAsRecommendedEvent.create(
        prompt="test",
        base_model="SDXL",
        lora_ids=[],
        recommendation_count=0,
    )

    metadata = await bus.publish(event)

    # One handler failed, one succeeded
    assert metadata.handler_count == 2
    assert not metadata.success  # Failed because one handler failed
    assert len(metadata.failed_handlers) == 1
    assert "FailingHandler" in metadata.failed_handlers[0]

    # Successful handler should still have been called
    assert len(successful_handler_called) == 1


@pytest.mark.asyncio
async def test_event_bus_performance_monitoring():
    """Test performance monitoring in event bus."""
    bus = EventBus(enable_metrics=True)
    handler = LoggingEventHandler()
    bus.subscribe(LoRAsRecommendedEvent, handler)

    event = LoRAsRecommendedEvent.create(
        prompt="test",
        base_model="SDXL",
        lora_ids=[],
        recommendation_count=0,
    )

    metadata = await bus.publish(event)

    assert metadata.processing_time_ms >= 0
    assert isinstance(metadata.processing_time_ms, float)


@pytest.mark.asyncio
async def test_event_bus_get_all_event_types():
    """Test getting all registered event types."""
    bus = EventBus()

    bus.subscribe(LoRAsRecommendedEvent, LoggingEventHandler())
    bus.subscribe(ImageGeneratedEvent, MetricsEventHandler())

    event_types = bus.get_all_event_types()

    assert LoRAsRecommendedEvent in event_types
    assert ImageGeneratedEvent in event_types
    assert len(event_types) == 2


@pytest.mark.asyncio
async def test_event_bus_clear():
    """Test clearing all subscriptions."""
    bus = EventBus()

    bus.subscribe(LoRAsRecommendedEvent, LoggingEventHandler())
    bus.subscribe(ImageGeneratedEvent, MetricsEventHandler())

    assert len(bus.get_all_event_types()) == 2

    bus.clear()

    assert len(bus.get_all_event_types()) == 0


# ============================================================================
# LoRA Event Tests
# ============================================================================

def test_loras_recommended_event_creation():
    """Test LoRAsRecommendedEvent creation."""
    event = LoRAsRecommendedEvent.create(
        prompt="anime girl",
        base_model="SDXL",
        lora_ids=["lora1", "lora2"],
        recommendation_count=2,
        confidence_threshold=0.7,
    )

    assert event.prompt == "anime girl"
    assert event.base_model == "SDXL"
    assert event.lora_ids == ["lora1", "lora2"]
    assert event.recommendation_count == 2
    assert event.confidence_threshold == 0.7
    assert event.event_id is not None
    assert event.occurred_at is not None


def test_loras_recommended_event_immutability():
    """Test that events are immutable."""
    event = LoRAsRecommendedEvent.create(
        prompt="test",
        base_model="SDXL",
        lora_ids=[],
        recommendation_count=0,
    )

    with pytest.raises(Exception):  # FrozenInstanceError
        event.prompt = "modified"


def test_top_lora_recommended_event_creation():
    """Test TopLoRARecommendedEvent creation."""
    event = TopLoRARecommendedEvent.create(
        prompt="cyberpunk city",
        base_model="SDXL",
        lora_id="cyberpunk-lora",
        confidence=0.95,
    )

    assert event.prompt == "cyberpunk city"
    assert event.base_model == "SDXL"
    assert event.lora_id == "cyberpunk-lora"
    assert event.confidence == 0.95


def test_lora_loaded_event_creation():
    """Test LoRALoadedEvent creation."""
    event = LoRALoadedEvent.create(
        lora_id="anime-style-v1",
        lora_name="Anime Style V1",
        base_model="SDXL",
    )

    assert event.lora_id == "anime-style-v1"
    assert event.lora_name == "Anime Style V1"
    assert event.base_model == "SDXL"


def test_lora_filtered_event_creation():
    """Test LoRAFilteredEvent creation."""
    event = LoRAFilteredEvent.create(
        original_count=10,
        filtered_count=3,
        confidence_threshold=0.7,
        base_model="SDXL",
    )

    assert event.original_count == 10
    assert event.filtered_count == 3
    assert event.confidence_threshold == 0.7
    assert event.base_model == "SDXL"


# ============================================================================
# Image Event Tests
# ============================================================================

def test_image_generation_requested_event_creation():
    """Test ImageGenerationRequestedEvent creation."""
    event = ImageGenerationRequestedEvent.create(
        prompt="anime girl",
        negative_prompt="bad quality",
        base_model="SDXL",
        loras_requested=["lora1", "lora2"],
        seed=42,
    )

    assert event.prompt == "anime girl"
    assert event.negative_prompt == "bad quality"
    assert event.base_model == "SDXL"
    assert event.loras_requested == ["lora1", "lora2"]
    assert event.seed == 42


def test_image_generated_event_creation():
    """Test ImageGeneratedEvent creation."""
    event = ImageGeneratedEvent.create(
        image_path="/output/image.png",
        prompt="anime girl",
        base_model="SDXL",
        loras_used=["lora1"],
        generation_time_seconds=12.5,
        seed=42,
    )

    assert event.image_path == "/output/image.png"
    assert event.prompt == "anime girl"
    assert event.base_model == "SDXL"
    assert event.loras_used == ["lora1"]
    assert event.generation_time_seconds == 12.5
    assert event.seed == 42


def test_image_generation_failed_event_creation():
    """Test ImageGenerationFailedEvent creation."""
    event = ImageGenerationFailedEvent.create(
        prompt="anime girl",
        base_model="SDXL",
        error_message="Out of memory",
        error_type="RuntimeError",
        loras_attempted=["heavy-lora"],
    )

    assert event.prompt == "anime girl"
    assert event.base_model == "SDXL"
    assert event.error_message == "Out of memory"
    assert event.error_type == "RuntimeError"
    assert event.loras_attempted == ["heavy-lora"]


def test_prompt_analyzed_event_creation():
    """Test PromptAnalyzedEvent creation."""
    event = PromptAnalyzedEvent.create(
        original_prompt="girl with blue hair",
        optimized_prompt="anime girl, blue hair, detailed",
        detected_intent="character_portrait",
        complexity_score=0.6,
        token_count=45,
    )

    assert event.original_prompt == "girl with blue hair"
    assert event.optimized_prompt == "anime girl, blue hair, detailed"
    assert event.detected_intent == "character_portrait"
    assert event.complexity_score == 0.6
    assert event.token_count == 45


# ============================================================================
# Event Handler Tests
# ============================================================================

@pytest.mark.asyncio
async def test_logging_event_handler():
    """Test LoggingEventHandler."""
    handler = LoggingEventHandler()

    event = LoRAsRecommendedEvent.create(
        prompt="test",
        base_model="SDXL",
        lora_ids=["lora1"],
        recommendation_count=1,
    )

    # Should not raise
    await handler.handle(event)


@pytest.mark.asyncio
async def test_metrics_event_handler():
    """Test MetricsEventHandler tracks metrics."""
    handler = MetricsEventHandler()

    assert handler.total_images_generated == 0

    event1 = ImageGeneratedEvent.create(
        image_path="/img1.png",
        prompt="test1",
        base_model="SDXL",
        loras_used=[],
        generation_time_seconds=10.0,
        seed=1,
    )

    event2 = ImageGeneratedEvent.create(
        image_path="/img2.png",
        prompt="test2",
        base_model="SDXL",
        loras_used=[],
        generation_time_seconds=20.0,
        seed=2,
    )

    await handler.handle(event1)
    await handler.handle(event2)

    assert handler.total_images_generated == 2
    assert handler.total_generation_time == 30.0
    assert handler.get_average_generation_time() == 15.0


@pytest.mark.asyncio
async def test_metrics_handler_tracks_by_model():
    """Test MetricsEventHandler tracks images by model."""
    handler = MetricsEventHandler()

    sdxl_event = ImageGeneratedEvent.create(
        image_path="/img.png",
        prompt="test",
        base_model="SDXL",
        loras_used=[],
        generation_time_seconds=10.0,
        seed=1,
    )

    sd15_event = ImageGeneratedEvent.create(
        image_path="/img2.png",
        prompt="test",
        base_model="SD 1.5",
        loras_used=[],
        generation_time_seconds=10.0,
        seed=2,
    )

    await handler.handle(sdxl_event)
    await handler.handle(sdxl_event)
    await handler.handle(sd15_event)

    assert handler.images_by_model["SDXL"] == 2
    assert handler.images_by_model["SD 1.5"] == 1


@pytest.mark.asyncio
async def test_error_logging_handler():
    """Test ErrorLoggingHandler tracks errors."""
    handler = ErrorLoggingHandler()

    assert handler.error_count == 0

    event1 = ImageGenerationFailedEvent.create(
        prompt="test",
        base_model="SDXL",
        error_message="Out of memory",
        error_type="RuntimeError",
    )

    event2 = ImageGenerationFailedEvent.create(
        prompt="test",
        base_model="SDXL",
        error_message="Invalid input",
        error_type="ValueError",
    )

    await handler.handle(event1)
    await handler.handle(event2)

    assert handler.error_count == 2
    assert handler.errors_by_type["RuntimeError"] == 1
    assert handler.errors_by_type["ValueError"] == 1


@pytest.mark.asyncio
async def test_caching_handler():
    """Test CachingHandler tracks load counts."""
    handler = CachingHandler()

    event = LoRALoadedEvent.create(
        lora_id="anime-v1",
        lora_name="Anime Style V1",
        base_model="SDXL",
    )

    # Load same LoRA multiple times
    for _ in range(5):
        await handler.handle(event)

    assert handler.load_counts["anime-v1"] == 5

    most_loaded = handler.get_most_loaded_loras(top_n=1)
    assert most_loaded[0] == ("anime-v1", 5)


@pytest.mark.asyncio
async def test_caching_handler_most_loaded():
    """Test CachingHandler returns most loaded LoRAs."""
    handler = CachingHandler()

    # Load different LoRAs different number of times
    loras = [
        ("lora1", 5),
        ("lora2", 10),
        ("lora3", 3),
        ("lora4", 7),
    ]

    for lora_id, count in loras:
        for _ in range(count):
            event = LoRALoadedEvent.create(
                lora_id=lora_id,
                lora_name=f"LoRA {lora_id}",
                base_model="SDXL",
            )
            await handler.handle(event)

    top_3 = handler.get_most_loaded_loras(top_n=3)

    assert len(top_3) == 3
    assert top_3[0] == ("lora2", 10)  # Most loaded
    assert top_3[1] == ("lora4", 7)
    assert top_3[2] == ("lora1", 5)


@pytest.mark.asyncio
async def test_prompt_analytics_handler():
    """Test PromptAnalyticsHandler tracks prompt patterns."""
    handler = PromptAnalyticsHandler()

    event1 = PromptAnalyzedEvent.create(
        original_prompt="test1",
        optimized_prompt="optimized1",
        detected_intent="character_portrait",
        complexity_score=0.5,
        token_count=30,
    )

    event2 = PromptAnalyzedEvent.create(
        original_prompt="test2",
        optimized_prompt="optimized2",
        detected_intent="landscape",
        complexity_score=0.7,
        token_count=40,
    )

    await handler.handle(event1)
    await handler.handle(event2)

    assert handler.prompt_count == 2
    assert handler.get_average_complexity() == 0.6  # (0.5 + 0.7) / 2

    intent_dist = handler.get_intent_distribution()
    assert intent_dist["character_portrait"] == 50.0
    assert intent_dist["landscape"] == 50.0


@pytest.mark.asyncio
async def test_multi_event_handler():
    """Test MultiEventHandler handles multiple event types."""
    handler = MultiEventHandler()

    success_event = ImageGeneratedEvent.create(
        image_path="/img.png",
        prompt="test",
        base_model="SDXL",
        loras_used=[],
        generation_time_seconds=10.0,
        seed=1,
    )

    failure_event = ImageGenerationFailedEvent.create(
        prompt="test",
        base_model="SDXL",
        error_message="Error",
        error_type="RuntimeError",
    )

    await handler.handle(success_event)
    await handler.handle(failure_event)

    assert handler.success_count == 1
    assert handler.failure_count == 1
    assert handler.get_success_rate() == 50.0  # 1 success, 1 failure


# ============================================================================
# EventMetadata Tests
# ============================================================================

def test_event_metadata_creation():
    """Test EventMetadata creation."""
    metadata = EventMetadata(
        event_type="TestEvent",
        handler_count=3,
        processing_time_ms=12.5,
        failed_handlers=["Handler1"],
        success=False,
    )

    assert metadata.event_type == "TestEvent"
    assert metadata.handler_count == 3
    assert metadata.processing_time_ms == 12.5
    assert metadata.failed_handlers == ["Handler1"]
    assert not metadata.success
    assert metadata.has_failures


def test_event_metadata_no_failures():
    """Test EventMetadata with no failures."""
    metadata = EventMetadata(
        event_type="TestEvent",
        handler_count=3,
        processing_time_ms=10.0,
        failed_handlers=[],
        success=True,
    )

    assert metadata.success
    assert not metadata.has_failures


# ============================================================================
# Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_full_event_workflow():
    """Test complete event-driven workflow."""
    bus = EventBus()

    # Setup handlers (MetricsEventHandler is designed for ImageGeneratedEvent)
    metrics_handler = MetricsEventHandler()

    bus.subscribe(ImageGeneratedEvent, metrics_handler)

    # Publish event
    event = ImageGeneratedEvent.create(
        image_path="/output/test.png",
        prompt="anime girl",
        base_model="SDXL",
        loras_used=["lora1"],
        generation_time_seconds=15.0,
        seed=42,
    )

    metadata = await bus.publish(event)

    assert metadata.success
    assert metadata.handler_count == 1
    assert metrics_handler.total_images_generated == 1
    assert metrics_handler.get_average_generation_time() == 15.0


@pytest.mark.asyncio
async def test_event_driven_monitoring_pipeline():
    """Test event-driven monitoring with multiple handlers."""
    bus = EventBus()

    # Setup monitoring pipeline
    metrics = MetricsEventHandler()
    errors = ErrorLoggingHandler()
    multi = MultiEventHandler()

    bus.subscribe(ImageGeneratedEvent, metrics)
    bus.subscribe(ImageGeneratedEvent, multi)
    bus.subscribe(ImageGenerationFailedEvent, errors)
    bus.subscribe(ImageGenerationFailedEvent, multi)

    # Simulate workflow
    success_event = ImageGeneratedEvent.create(
        image_path="/img.png",
        prompt="test",
        base_model="SDXL",
        loras_used=[],
        generation_time_seconds=10.0,
        seed=1,
    )

    failure_event = ImageGenerationFailedEvent.create(
        prompt="test",
        base_model="SDXL",
        error_message="Failed",
        error_type="RuntimeError",
    )

    await bus.publish(success_event)
    await bus.publish(failure_event)

    # Verify monitoring
    assert metrics.total_images_generated == 1
    assert errors.error_count == 1
    assert multi.success_count == 1
    assert multi.failure_count == 1
    assert multi.get_success_rate() == 50.0
