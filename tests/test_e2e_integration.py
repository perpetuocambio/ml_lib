"""End-to-End Integration Tests for Clean Architecture.

Tests the complete flow through all layers:
- Application Layer (Commands/Queries)
- Domain Layer (Services/Events)
- Infrastructure Layer (Repositories)

These tests validate that the entire CQRS + Event-Driven architecture
works correctly when all components are integrated.
"""

import pytest
import asyncio
from pathlib import Path
from typing import List

from ml_lib.diffusion.application.commands import (
    CommandBus,
    RecommendLoRAsCommand,
    RecommendTopLoRACommand,
    FilterConfidentRecommendationsCommand,
    RecommendLoRAsHandler,
    RecommendTopLoRAHandler,
    FilterConfidentRecommendationsHandler,
)
from ml_lib.diffusion.application.queries import (
    QueryBus,
    GetAllLoRAsQuery,
    GetLoRAsByBaseModelQuery,
    SearchLoRAsByPromptQuery,
    GetAllLoRAsHandler,
    GetLoRAsByBaseModelHandler,
    SearchLoRAsByPromptHandler,
)
from ml_lib.diffusion.domain.events import (
    EventBus,
    LoRAsRecommendedEvent,
    TopLoRARecommendedEvent,
    LoRALoadedEvent,
    LoRAFilteredEvent,
    ImageGenerationRequestedEvent,
    ImageGeneratedEvent,
    LoggingEventHandler,
    MetricsEventHandler,
    MultiEventHandler,
)
from ml_lib.diffusion.domain.entities.lora import LoRA
from ml_lib.diffusion.infrastructure.persistence.in_memory_model_repository import (
    InMemoryModelRepository,
)
from ml_lib.diffusion.domain.services.lora_recommendation_service import (
    LoRARecommendationService,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary directory for test data."""
    return tmp_path


@pytest.fixture
def populated_repository(test_data_dir):
    """Create repository with realistic test data."""
    repo = InMemoryModelRepository()

    # SDXL LoRAs - Anime/Manga style
    sdxl_anime_loras = [
        {
            "name": "anime_style_xl",
            "filename": "anime-style-xl.safetensors",
            "base_model": "SDXL",
            "weight": 0.85,
            "trigger_words": ["anime", "manga", "cel shading"],
            "tags": ["anime", "style", "manga"],
        },
        {
            "name": "detailed_anime_xl",
            "filename": "detailed-anime-xl.safetensors",
            "base_model": "SDXL",
            "weight": 0.75,
            "trigger_words": ["detailed anime", "high quality anime"],
            "tags": ["anime", "detailed", "quality"],
        },
    ]

    # SDXL LoRAs - Portrait/Character
    sdxl_portrait_loras = [
        {
            "name": "portrait_master_xl",
            "filename": "portrait-master-xl.safetensors",
            "base_model": "SDXL",
            "weight": 0.80,
            "trigger_words": ["portrait", "detailed face", "facial features"],
            "tags": ["portrait", "face", "character"],
        },
        {
            "name": "character_design_xl",
            "filename": "character-design-xl.safetensors",
            "base_model": "SDXL",
            "weight": 0.70,
            "trigger_words": ["character design", "character sheet"],
            "tags": ["character", "design", "reference"],
        },
    ]

    # SDXL LoRAs - Cyberpunk/SciFi
    sdxl_cyberpunk_loras = [
        {
            "name": "cyberpunk_xl",
            "filename": "cyberpunk-xl.safetensors",
            "base_model": "SDXL",
            "weight": 0.90,
            "trigger_words": ["cyberpunk", "neon", "futuristic city"],
            "tags": ["cyberpunk", "scifi", "neon"],
        },
    ]

    # Pony V6 LoRAs
    pony_loras = [
        {
            "name": "anime_pony_v6",
            "filename": "anime-pony-v6.safetensors",
            "base_model": "Pony Diffusion V6",
            "weight": 0.85,
            "trigger_words": ["anime", "score_9", "score_8_up"],
            "tags": ["anime", "pony", "quality"],
        },
        {
            "name": "cyberpunk_pony_v6",
            "filename": "cyberpunk-pony-v6.safetensors",
            "base_model": "Pony Diffusion V6",
            "weight": 0.88,
            "trigger_words": ["cyberpunk", "neon lights", "score_9"],
            "tags": ["cyberpunk", "pony", "scifi"],
        },
    ]

    # SD 1.5 LoRAs
    sd15_loras = [
        {
            "name": "anime_sd15",
            "filename": "anime-sd15.safetensors",
            "base_model": "SD 1.5",
            "weight": 0.75,
            "trigger_words": ["anime", "manga style"],
            "tags": ["anime", "sd15"],
        },
    ]

    # Add all LoRAs to repository
    all_loras = (
        sdxl_anime_loras
        + sdxl_portrait_loras
        + sdxl_cyberpunk_loras
        + pony_loras
        + sd15_loras
    )

    for lora_data in all_loras:
        # Create fake file
        lora_path = test_data_dir / lora_data["filename"]
        lora_path.write_text("fake lora content")

        lora = LoRA.create(
            name=lora_data["name"],
            path=lora_path,
            base_model=lora_data["base_model"],
            weight=lora_data["weight"],
            trigger_words=lora_data["trigger_words"],
            tags=lora_data["tags"],
        )
        repo.add_lora(lora)

    return repo


@pytest.fixture
def lora_service(populated_repository):
    """Create LoRA recommendation service."""
    return LoRARecommendationService(populated_repository)


@pytest.fixture
def event_bus():
    """Create event bus for testing."""
    return EventBus(enable_metrics=True)


@pytest.fixture
def command_bus(lora_service, event_bus):
    """Create command bus with all handlers registered."""
    bus = CommandBus()

    bus.register(
        RecommendLoRAsCommand,
        RecommendLoRAsHandler(lora_service, event_bus),
    )
    bus.register(
        RecommendTopLoRACommand,
        RecommendTopLoRAHandler(lora_service, event_bus),
    )
    bus.register(
        FilterConfidentRecommendationsCommand,
        FilterConfidentRecommendationsHandler(lora_service, event_bus),
    )

    return bus


@pytest.fixture
def query_bus(lora_service):
    """Create query bus with all handlers registered."""
    bus = QueryBus(enable_monitoring=True)

    bus.register(GetAllLoRAsQuery, GetAllLoRAsHandler(lora_service))
    bus.register(
        GetLoRAsByBaseModelQuery, GetLoRAsByBaseModelHandler(lora_service)
    )
    bus.register(
        SearchLoRAsByPromptQuery, SearchLoRAsByPromptHandler(lora_service)
    )

    return bus


# ============================================================================
# End-to-End Integration Tests - CQRS Workflows
# ============================================================================


def test_e2e_complete_cqrs_workflow(command_bus, query_bus, event_bus):
    """Test complete CQRS workflow: Query → Command → Event."""
    # Step 1: Query - Browse all available LoRAs
    browse_query = GetAllLoRAsQuery()
    browse_result = query_bus.dispatch(browse_query)

    assert browse_result.data is not None
    assert len(browse_result.data) == 8  # Total LoRAs from fixture (8 created)
    assert browse_result.metadata["count"] == 8

    # Step 2: Query - Filter by base model (SDXL)
    filter_query = GetLoRAsByBaseModelQuery(base_model="SDXL")
    filter_result = query_bus.dispatch(filter_query)

    assert len(filter_result.data) == 7  # 5 SDXL + 2 Pony (SDXL-compatible)

    # Step 3: Command - Recommend LoRAs for a prompt
    recommend_cmd = RecommendLoRAsCommand(
        prompt="anime girl with detailed face and cyberpunk outfit",
        base_model="SDXL",
        max_loras=5,
        min_confidence=0.0,
    )

    recommend_result = command_bus.dispatch(recommend_cmd)

    assert recommend_result.is_success
    assert len(recommend_result.data) > 0
    assert recommend_result.metadata["count"] > 0

    # Step 4: Command - Filter confident recommendations
    filter_cmd = FilterConfidentRecommendationsCommand(
        recommendations=recommend_result.data
    )

    filter_cmd_result = command_bus.dispatch(filter_cmd)

    assert filter_cmd_result.is_success
    assert filter_cmd_result.metadata["original_count"] == len(
        recommend_result.data
    )

    # Step 5: Verify final recommendations are high quality
    for rec in filter_cmd_result.data:
        assert rec.confidence >= 0.5  # CONFIDENCE_THRESHOLD


def test_e2e_multi_model_workflow(command_bus, query_bus):
    """Test workflow comparing recommendations across different base models."""
    prompt = "anime character portrait"

    results = {}

    for base_model in ["SDXL", "Pony Diffusion V6", "SD 1.5"]:
        # Query: Get LoRAs for this model
        query = GetLoRAsByBaseModelQuery(base_model=base_model)
        query_result = query_bus.dispatch(query)

        # Command: Get recommendations for this model
        command = RecommendLoRAsCommand(
            prompt=prompt,
            base_model=base_model,
            max_loras=3,
            min_confidence=0.0,
        )
        command_result = command_bus.dispatch(command)

        results[base_model] = {
            "available_loras": len(query_result.data),
            "recommended_loras": len(command_result.data),
        }

    # Verify each model has appropriate LoRAs
    assert results["SDXL"]["available_loras"] >= 5
    assert results["Pony Diffusion V6"]["available_loras"] >= 2
    assert results["SD 1.5"]["available_loras"] >= 1

    # Verify recommendations are model-specific
    assert results["SDXL"]["recommended_loras"] > 0
    assert results["Pony Diffusion V6"]["recommended_loras"] > 0


@pytest.mark.asyncio
async def test_e2e_event_driven_workflow(
    command_bus, query_bus, event_bus, lora_service
):
    """Test complete event-driven workflow with monitoring."""
    # Setup event handlers for monitoring
    metrics_handler = MetricsEventHandler()
    logging_handler = LoggingEventHandler()
    multi_handler = MultiEventHandler()

    # Subscribe handlers
    event_bus.subscribe(LoRAsRecommendedEvent, metrics_handler)
    event_bus.subscribe(LoRAsRecommendedEvent, logging_handler)
    event_bus.subscribe(TopLoRARecommendedEvent, multi_handler)
    event_bus.subscribe(ImageGeneratedEvent, metrics_handler)
    event_bus.subscribe(ImageGeneratedEvent, multi_handler)

    # Step 1: Execute command that triggers event
    recommend_cmd = RecommendLoRAsCommand(
        prompt="cyberpunk anime girl",
        base_model="SDXL",
        max_loras=3,
    )

    recommend_result = command_bus.dispatch(recommend_cmd)
    assert recommend_result.is_success

    # Give async handlers time to process (fire-and-forget pattern)
    await asyncio.sleep(0.1)

    # Step 2: Get top recommendation
    top_cmd = RecommendTopLoRACommand(
        prompt="cyberpunk anime girl",
        base_model="SDXL",
    )

    top_result = command_bus.dispatch(top_cmd)
    assert top_result.is_success

    await asyncio.sleep(0.1)

    # Step 3: Simulate image generation event
    generation_event = ImageGeneratedEvent.create(
        image_path="/output/test.png",
        prompt="cyberpunk anime girl",
        base_model="SDXL",
        loras_used=["cyberpunk_xl", "anime_style_xl"],
        generation_time_seconds=15.5,
        seed=42,
    )

    metadata = await event_bus.publish(generation_event)

    assert metadata.success
    assert metadata.handler_count >= 1

    # Verify metrics were collected
    assert metrics_handler.total_images_generated >= 1
    assert multi_handler.success_count >= 1


def test_e2e_search_and_recommend_workflow(query_bus, command_bus):
    """Test workflow: search by prompt → get recommendations."""
    # Use a simpler, more direct prompt that will match our test data
    prompt = "anime girl"

    # Step 1: Search for LoRAs matching keywords
    search_query = SearchLoRAsByPromptQuery(
        prompt=prompt,
        base_model="SDXL",
    )

    search_result = query_bus.dispatch(search_query)

    assert search_result.data is not None
    assert len(search_result.data) > 0, f"Search should find anime LoRAs, found {len(search_result.data)}"

    # Step 2: Get detailed recommendations for the same prompt
    recommend_cmd = RecommendLoRAsCommand(
        prompt=prompt,
        base_model="SDXL",
        max_loras=5,
        min_confidence=0.0,  # Accept all recommendations for testing
    )

    recommend_result = command_bus.dispatch(recommend_cmd)

    assert recommend_result.is_success
    # Both workflows should return results for this straightforward prompt
    assert len(search_result.data) > 0, "Search should find matching LoRAs"
    assert len(recommend_result.data) > 0, "Recommend should return results"


def test_e2e_filter_pipeline(command_bus, query_bus):
    """Test complete filtering pipeline through multiple stages."""
    prompt = "anime character design"

    # Stage 1: Get all LoRAs
    all_query = GetAllLoRAsQuery()
    all_result = query_bus.dispatch(all_query)
    total_loras = len(all_result.data)

    # Stage 2: Filter by base model
    model_query = GetLoRAsByBaseModelQuery(base_model="SDXL")
    model_result = query_bus.dispatch(model_query)
    sdxl_loras = len(model_result.data)

    assert sdxl_loras < total_loras

    # Stage 3: Search by prompt within model
    search_query = SearchLoRAsByPromptQuery(
        prompt=prompt,
        base_model="SDXL",
    )
    search_result = query_bus.dispatch(search_query)
    matching_loras = len(search_result.data)

    assert matching_loras <= sdxl_loras

    # Stage 4: Get recommendations (most refined)
    recommend_cmd = RecommendLoRAsCommand(
        prompt=prompt,
        base_model="SDXL",
        max_loras=3,
    )
    recommend_result = command_bus.dispatch(recommend_cmd)

    assert recommend_result.is_success
    assert len(recommend_result.data) <= matching_loras

    # Stage 5: Filter only confident recommendations
    filter_cmd = FilterConfidentRecommendationsCommand(
        recommendations=recommend_result.data
    )
    filter_result = command_bus.dispatch(filter_cmd)

    assert filter_result.is_success
    assert len(filter_result.data) <= len(recommend_result.data)


# ============================================================================
# End-to-End Integration Tests - Error Handling
# ============================================================================


def test_e2e_error_handling_validation(command_bus):
    """Test end-to-end validation error handling."""
    # Invalid command: empty prompt
    invalid_cmd = RecommendLoRAsCommand(
        prompt="",
        base_model="SDXL",
        max_loras=3,
    )

    result = command_bus.dispatch(invalid_cmd)

    assert not result.is_success
    assert result.error is not None
    assert "Prompt cannot be empty" in result.error


def test_e2e_error_handling_not_found(command_bus):
    """Test end-to-end not found error handling."""
    # Command that won't find matches
    cmd = RecommendTopLoRACommand(
        prompt="nonexistent_style_xyz_abc_123",
        base_model="NonExistentModel",
    )

    result = command_bus.dispatch(cmd)

    assert not result.is_success
    # May be validation error or not found depending on implementation


@pytest.mark.asyncio
async def test_e2e_event_error_isolation(event_bus):
    """Test that event handler errors don't break the workflow."""
    from ml_lib.diffusion.domain.events.base import IEventHandler

    # Create a failing handler
    class FailingHandler(IEventHandler[ImageGeneratedEvent]):
        async def handle(self, event: ImageGeneratedEvent) -> None:
            raise RuntimeError("Handler failed intentionally")

    # Create a successful handler
    successful_calls = []

    class SuccessfulHandler(IEventHandler[ImageGeneratedEvent]):
        async def handle(self, event: ImageGeneratedEvent) -> None:
            successful_calls.append(event)

    # Subscribe both handlers
    event_bus.subscribe(ImageGeneratedEvent, FailingHandler())
    event_bus.subscribe(ImageGeneratedEvent, SuccessfulHandler())

    # Publish event
    event = ImageGeneratedEvent.create(
        image_path="/test.png",
        prompt="test",
        base_model="SDXL",
        loras_used=[],
        generation_time_seconds=10.0,
        seed=1,
    )

    metadata = await event_bus.publish(event)

    # Event publishing should complete despite error
    assert metadata.handler_count == 2
    assert not metadata.success  # Failed because one handler failed
    assert len(metadata.failed_handlers) == 1

    # Successful handler should still have been called
    assert len(successful_calls) == 1


# ============================================================================
# End-to-End Integration Tests - Performance
# ============================================================================


def test_e2e_performance_bulk_queries(query_bus):
    """Test performance with bulk query operations."""
    import time

    # Execute multiple queries rapidly
    start_time = time.time()

    for _ in range(10):
        query = GetAllLoRAsQuery()
        result = query_bus.dispatch(query)
        assert result.data is not None

    elapsed = time.time() - start_time

    # Should complete quickly (< 1 second for 10 queries)
    assert elapsed < 1.0


def test_e2e_performance_bulk_commands(command_bus):
    """Test performance with bulk command operations."""
    import time

    prompts = [
        "anime girl",
        "cyberpunk city",
        "portrait",
        "character design",
        "detailed face",
    ]

    start_time = time.time()

    for prompt in prompts:
        cmd = RecommendLoRAsCommand(
            prompt=prompt,
            base_model="SDXL",
            max_loras=3,
        )
        result = command_bus.dispatch(cmd)
        assert result.is_success

    elapsed = time.time() - start_time

    # Should complete reasonably quickly (< 2 seconds for 5 commands)
    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_e2e_performance_concurrent_events(event_bus):
    """Test performance with concurrent event publishing."""
    import time

    handler = MetricsEventHandler()
    event_bus.subscribe(ImageGeneratedEvent, handler)

    start_time = time.time()

    # Publish 20 events concurrently
    tasks = []
    for i in range(20):
        event = ImageGeneratedEvent.create(
            image_path=f"/img{i}.png",
            prompt=f"test{i}",
            base_model="SDXL",
            loras_used=[],
            generation_time_seconds=10.0,
            seed=i,
        )
        tasks.append(event_bus.publish(event))

    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    # All should succeed
    assert all(r.success for r in results)

    # Should complete quickly (< 1 second for 20 events)
    assert elapsed < 1.0

    # Handler should have processed all events
    assert handler.total_images_generated == 20


# ============================================================================
# End-to-End Integration Tests - Data Consistency
# ============================================================================


def test_e2e_data_consistency_queries_are_readonly(
    query_bus, populated_repository
):
    """Test that queries don't modify repository state."""
    initial_count = len(populated_repository.get_all_loras())

    # Execute various queries
    query_bus.dispatch(GetAllLoRAsQuery())
    query_bus.dispatch(GetLoRAsByBaseModelQuery(base_model="SDXL"))
    query_bus.dispatch(
        SearchLoRAsByPromptQuery(prompt="anime", base_model="SDXL")
    )

    final_count = len(populated_repository.get_all_loras())

    # Repository should be unchanged
    assert final_count == initial_count


def test_e2e_data_consistency_command_results(command_bus):
    """Test that command results are consistent across multiple calls."""
    prompt = "anime portrait"
    base_model = "SDXL"

    # Execute same command multiple times
    results = []
    for _ in range(3):
        cmd = RecommendLoRAsCommand(
            prompt=prompt,
            base_model=base_model,
            max_loras=5,
        )
        result = command_bus.dispatch(cmd)
        results.append(result)

    # All should succeed
    assert all(r.is_success for r in results)

    # Results should be consistent (same LoRAs recommended)
    first_loras = {rec.lora.name for rec in results[0].data}
    for result in results[1:]:
        current_loras = {rec.lora.name for rec in result.data}
        assert current_loras == first_loras


@pytest.mark.asyncio
async def test_e2e_complete_image_generation_workflow(
    command_bus, query_bus, event_bus
):
    """Test complete realistic workflow: browse → recommend → generate → monitor."""
    # Setup monitoring (only for ImageGeneratedEvent to avoid AttributeError)
    metrics = MetricsEventHandler()
    multi = MultiEventHandler()

    event_bus.subscribe(ImageGeneratedEvent, metrics)
    event_bus.subscribe(ImageGeneratedEvent, multi)

    # Step 1: Browse available LoRAs
    browse_query = GetAllLoRAsQuery()
    browse_result = query_bus.dispatch(browse_query)
    assert len(browse_result.data) > 0

    # Step 2: Filter by desired model
    model_query = GetLoRAsByBaseModelQuery(base_model="SDXL")
    model_result = query_bus.dispatch(model_query)
    assert len(model_result.data) > 0

    # Step 3: Get recommendations for prompt
    prompt = "cyberpunk anime girl with neon lights"
    recommend_cmd = RecommendLoRAsCommand(
        prompt=prompt,
        base_model="SDXL",
        max_loras=3,
    )
    recommend_result = command_bus.dispatch(recommend_cmd)
    assert recommend_result.is_success

    # Step 4: Filter confident recommendations
    filter_cmd = FilterConfidentRecommendationsCommand(
        recommendations=recommend_result.data
    )
    filter_result = command_bus.dispatch(filter_cmd)
    assert filter_result.is_success

    # Step 5: Simulate image generation request
    request_event = ImageGenerationRequestedEvent.create(
        prompt=prompt,
        negative_prompt="low quality, blurry",
        base_model="SDXL",
        loras_requested=[rec.lora.name for rec in filter_result.data],
        seed=42,
    )
    await event_bus.publish(request_event)

    # Step 6: Simulate successful generation
    generation_event = ImageGeneratedEvent.create(
        image_path="/output/cyberpunk_anime_girl.png",
        prompt=prompt,
        base_model="SDXL",
        loras_used=[rec.lora.name for rec in filter_result.data],
        generation_time_seconds=18.5,
        seed=42,
    )
    metadata = await event_bus.publish(generation_event)

    await asyncio.sleep(0.1)

    # Verify complete workflow
    assert metadata.success
    assert metrics.total_images_generated == 1
    assert multi.success_count == 1
    assert multi.get_success_rate() == 100.0
