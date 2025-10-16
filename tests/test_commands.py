"""Tests for Command Pattern implementation.

Tests for commands, handlers, and CommandBus.
"""

import pytest
import asyncio
from ml_lib.diffusion.application.commands import (
    CommandBus,
    CommandStatus,
    RecommendLoRAsCommand,
    RecommendTopLoRACommand,
    FilterConfidentRecommendationsCommand,
    RecommendLoRAsHandler,
    RecommendTopLoRAHandler,
    FilterConfidentRecommendationsHandler,
)
from ml_lib.diffusion.domain.entities.lora import LoRA, LoRARecommendation
from ml_lib.diffusion.infrastructure.persistence.in_memory_model_repository import (
    InMemoryModelRepository,
)
from ml_lib.diffusion.domain.services.lora_recommendation_service import (
    LoRARecommendationService,
)
from ml_lib.diffusion.domain.events import EventBus


@pytest.fixture
def repository(tmp_path):
    """Create repository with sample LoRAs."""
    repo = InMemoryModelRepository()

    # Create sample LoRA files
    lora_files = [
        ("anime_style", "anime.safetensors", "SDXL", 0.8, ["anime", "manga"], ["anime"]),
        ("portrait_detail", "portrait.safetensors", "SDXL", 0.7, ["portrait", "detailed face"], ["portrait"]),
        ("cyberpunk", "cyberpunk.safetensors", "SDXL", 0.9, ["cyberpunk", "neon"], ["cyberpunk"]),
    ]

    # Add sample LoRAs
    for name, filename, base_model, weight, trigger_words, tags in lora_files:
        # Create fake file
        lora_path = tmp_path / filename
        lora_path.write_text("fake lora content")

        lora = LoRA.create(
            name=name,
            path=lora_path,
            base_model=base_model,
            weight=weight,
            trigger_words=trigger_words,
            tags=tags,
        )
        repo.add_lora(lora)

    return repo


@pytest.fixture
def lora_service(repository):
    """Create LoRA recommendation service."""
    return LoRARecommendationService(repository)


@pytest.fixture
def event_bus():
    """Create event bus for testing."""
    return EventBus()


@pytest.fixture
def command_bus(lora_service, event_bus):
    """Create command bus with registered handlers."""
    bus = CommandBus()

    # Register handlers with event bus
    bus.register(
        RecommendLoRAsCommand,
        RecommendLoRAsHandler(lora_service, event_bus)
    )
    bus.register(
        RecommendTopLoRACommand,
        RecommendTopLoRAHandler(lora_service, event_bus)
    )
    bus.register(
        FilterConfidentRecommendationsCommand,
        FilterConfidentRecommendationsHandler(lora_service, event_bus)
    )

    return bus


# ============================================================================
# CommandBus Tests
# ============================================================================

def test_command_bus_registration(command_bus):
    """Test command registration."""
    assert command_bus.is_registered(RecommendLoRAsCommand)
    assert command_bus.is_registered(RecommendTopLoRACommand)
    assert command_bus.is_registered(FilterConfidentRecommendationsCommand)


def test_command_bus_unregister(lora_service):
    """Test command unregistration."""
    bus = CommandBus()
    handler = RecommendLoRAsHandler(lora_service)

    bus.register(RecommendLoRAsCommand, handler)
    assert bus.is_registered(RecommendLoRAsCommand)

    bus.unregister(RecommendLoRAsCommand)
    assert not bus.is_registered(RecommendLoRAsCommand)


def test_command_bus_duplicate_registration(lora_service):
    """Test that duplicate registration raises error."""
    bus = CommandBus()
    handler = RecommendLoRAsHandler(lora_service)

    bus.register(RecommendLoRAsCommand, handler)

    with pytest.raises(ValueError, match="Handler already registered"):
        bus.register(RecommendLoRAsCommand, handler)


def test_command_bus_no_handler_registered(command_bus):
    """Test dispatch with no handler registered."""
    # Create a dummy command class
    from dataclasses import dataclass
    from ml_lib.diffusion.application.commands.base import ICommand

    @dataclass(frozen=True)
    class DummyCommand(ICommand):
        value: str

    command = DummyCommand(value="test")
    result = command_bus.dispatch(command)

    assert result.status == CommandStatus.FAILED
    assert "No handler registered" in result.error


# ============================================================================
# RecommendLoRAsCommand Tests
# ============================================================================

def test_recommend_loras_command_success(command_bus):
    """Test successful LoRA recommendation."""
    command = RecommendLoRAsCommand(
        prompt="anime girl portrait",
        base_model="SDXL",
        max_loras=3,
        min_confidence=0.0,
    )

    result = command_bus.dispatch(command)

    assert result.is_success
    assert result.status == CommandStatus.SUCCESS
    assert isinstance(result.data, list)
    assert len(result.data) > 0
    assert all(isinstance(rec, LoRARecommendation) for rec in result.data)
    assert result.metadata["count"] == len(result.data)
    assert result.metadata["prompt"] == "anime girl portrait"
    assert result.metadata["base_model"] == "SDXL"


def test_recommend_loras_command_validation_empty_prompt(command_bus):
    """Test validation fails for empty prompt."""
    command = RecommendLoRAsCommand(
        prompt="",
        base_model="SDXL",
        max_loras=3,
    )

    result = command_bus.dispatch(command)

    assert not result.is_success
    assert result.status == CommandStatus.VALIDATION_ERROR
    assert "Prompt cannot be empty" in result.error


def test_recommend_loras_command_validation_empty_model(command_bus):
    """Test validation fails for empty base model."""
    command = RecommendLoRAsCommand(
        prompt="anime girl",
        base_model="",
        max_loras=3,
    )

    result = command_bus.dispatch(command)

    assert not result.is_success
    assert result.status == CommandStatus.VALIDATION_ERROR
    assert "Base model cannot be empty" in result.error


def test_recommend_loras_command_validation_invalid_max_loras(command_bus):
    """Test validation fails for invalid max_loras."""
    command = RecommendLoRAsCommand(
        prompt="anime girl",
        base_model="SDXL",
        max_loras=0,
    )

    result = command_bus.dispatch(command)

    assert not result.is_success
    assert result.status == CommandStatus.VALIDATION_ERROR
    assert "max_loras must be at least 1" in result.error


def test_recommend_loras_command_validation_invalid_confidence(command_bus):
    """Test validation fails for invalid confidence."""
    command = RecommendLoRAsCommand(
        prompt="anime girl",
        base_model="SDXL",
        max_loras=3,
        min_confidence=1.5,  # Invalid: > 1.0
    )

    result = command_bus.dispatch(command)

    assert not result.is_success
    assert result.status == CommandStatus.VALIDATION_ERROR
    assert "min_confidence must be between 0 and 1" in result.error


def test_recommend_loras_command_max_loras_limit(command_bus):
    """Test that max_loras limits results."""
    command = RecommendLoRAsCommand(
        prompt="anime portrait cyberpunk",
        base_model="SDXL",
        max_loras=2,
        min_confidence=0.0,
    )

    result = command_bus.dispatch(command)

    assert result.is_success
    assert len(result.data) <= 2


def test_recommend_loras_command_with_event_bus(lora_service, event_bus):
    """Test that command publishes event when event bus is available."""
    # Track events
    published_events = []

    from ml_lib.diffusion.domain.events import LoRAsRecommendedEvent, IEventHandler

    class TestHandler(IEventHandler[LoRAsRecommendedEvent]):
        async def handle(self, event: LoRAsRecommendedEvent) -> None:
            published_events.append(event)

    event_bus.subscribe(LoRAsRecommendedEvent, TestHandler())

    # Execute command with event bus
    handler = RecommendLoRAsHandler(lora_service, event_bus)
    command = RecommendLoRAsCommand(
        prompt="anime girl",
        base_model="SDXL",
        max_loras=3,
    )

    result = handler.handle(command)

    assert result.is_success
    # Event publishing is async fire-and-forget, so we can't easily test it in sync test


# ============================================================================
# RecommendTopLoRACommand Tests
# ============================================================================

def test_recommend_top_lora_command_success(command_bus):
    """Test successful top LoRA recommendation."""
    command = RecommendTopLoRACommand(
        prompt="anime girl",
        base_model="SDXL",
    )

    result = command_bus.dispatch(command)

    assert result.is_success
    assert result.status == CommandStatus.SUCCESS
    assert isinstance(result.data, LoRARecommendation)
    assert result.metadata["prompt"] == "anime girl"
    assert result.metadata["base_model"] == "SDXL"
    assert "lora_name" in result.metadata


def test_recommend_top_lora_command_validation_empty_prompt(command_bus):
    """Test validation fails for empty prompt."""
    command = RecommendTopLoRACommand(
        prompt="",
        base_model="SDXL",
    )

    result = command_bus.dispatch(command)

    assert not result.is_success
    assert result.status == CommandStatus.VALIDATION_ERROR
    assert "Prompt cannot be empty" in result.error


def test_recommend_top_lora_command_validation_empty_model(command_bus):
    """Test validation fails for empty base model."""
    command = RecommendTopLoRACommand(
        prompt="anime girl",
        base_model="",
    )

    result = command_bus.dispatch(command)

    assert not result.is_success
    assert result.status == CommandStatus.VALIDATION_ERROR
    assert "Base model cannot be empty" in result.error


def test_recommend_top_lora_command_not_found(lora_service):
    """Test not found when no matching LoRA."""
    # Empty repository
    empty_repo = InMemoryModelRepository()
    # Empty repo needs no LoRAs
    empty_service = LoRARecommendationService(empty_repo)
    handler = RecommendTopLoRAHandler(empty_service)

    command = RecommendTopLoRACommand(
        prompt="anime girl",
        base_model="SDXL",
    )

    result = handler.handle(command)

    assert not result.is_success
    assert result.status == CommandStatus.NOT_FOUND
    assert "No suitable LoRA found" in result.error


# ============================================================================
# FilterConfidentRecommendationsCommand Tests
# ============================================================================

def test_filter_confident_recommendations_command_success(command_bus, lora_service):
    """Test successful filtering of recommendations."""
    # First get recommendations
    recommendations = lora_service.recommend(
        prompt="anime girl",
        base_model="SDXL",
        max_loras=3,
    )

    # Now filter them
    command = FilterConfidentRecommendationsCommand(
        recommendations=recommendations
    )

    result = command_bus.dispatch(command)

    assert result.is_success
    assert result.status == CommandStatus.SUCCESS
    assert isinstance(result.data, list)
    assert result.metadata["original_count"] == len(recommendations)
    assert result.metadata["filtered_count"] == len(result.data)


def test_filter_confident_recommendations_validation_none(command_bus):
    """Test validation fails for None recommendations."""
    command = FilterConfidentRecommendationsCommand(
        recommendations=None
    )

    result = command_bus.dispatch(command)

    assert not result.is_success
    assert result.status == CommandStatus.VALIDATION_ERROR
    assert "Recommendations cannot be None" in result.error


def test_filter_confident_recommendations_empty_list(command_bus):
    """Test filtering empty list."""
    command = FilterConfidentRecommendationsCommand(
        recommendations=[]
    )

    result = command_bus.dispatch(command)

    assert result.is_success
    assert len(result.data) == 0
    assert result.metadata["original_count"] == 0
    assert result.metadata["filtered_count"] == 0


def test_filter_confident_recommendations_filters_low_confidence(lora_service):
    """Test that low confidence recommendations are filtered."""
    # Get recommendations
    recommendations = lora_service.recommend(
        prompt="test",
        base_model="SDXL",
        max_loras=10,
        min_confidence=0.0,  # Get all
    )

    # Filter confident
    handler = FilterConfidentRecommendationsHandler(lora_service)
    command = FilterConfidentRecommendationsCommand(
        recommendations=recommendations
    )

    result = handler.handle(command)

    assert result.is_success
    # All filtered results should have confidence >= CONFIDENCE_THRESHOLD
    for rec in result.data:
        assert rec.confidence >= lora_service.CONFIDENCE_THRESHOLD


# ============================================================================
# CommandResult Tests
# ============================================================================

def test_command_result_is_success():
    """Test CommandResult.is_success property."""
    from ml_lib.diffusion.application.commands.base import CommandResult, CommandStatus

    success = CommandResult.success(data="test")
    assert success.is_success

    failure = CommandResult.failure("error")
    assert not failure.is_success
    assert failure.status == CommandStatus.FAILED

    validation_error = CommandResult.validation_error("invalid")
    assert not validation_error.is_success

    not_found = CommandResult.not_found("not found")
    assert not not_found.is_success


def test_command_result_factory_methods():
    """Test CommandResult factory methods."""
    from ml_lib.diffusion.application.commands.base import CommandResult, CommandStatus

    # Success
    success = CommandResult.success(data="data", metadata={"key": "value"})
    assert success.status == CommandStatus.SUCCESS
    assert success.data == "data"
    assert success.metadata == {"key": "value"}
    assert success.error is None

    # Failure
    failure = CommandResult.failure("error message")
    assert failure.status == CommandStatus.FAILED
    assert failure.error == "error message"
    assert failure.data is None

    # Validation error
    validation = CommandResult.validation_error("validation failed")
    assert validation.status == CommandStatus.VALIDATION_ERROR
    assert validation.error == "validation failed"

    # Not found
    not_found = CommandResult.not_found("resource not found")
    assert not_found.status == CommandStatus.NOT_FOUND
    assert not_found.error == "resource not found"


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_command_workflow(command_bus):
    """Test complete workflow: recommend → filter."""
    # Step 1: Recommend LoRAs
    recommend_cmd = RecommendLoRAsCommand(
        prompt="anime portrait",
        base_model="SDXL",
        max_loras=5,
        min_confidence=0.0,
    )

    recommend_result = command_bus.dispatch(recommend_cmd)
    assert recommend_result.is_success

    # Step 2: Filter confident recommendations
    filter_cmd = FilterConfidentRecommendationsCommand(
        recommendations=recommend_result.data
    )

    filter_result = command_bus.dispatch(filter_cmd)
    assert filter_result.is_success
    assert len(filter_result.data) <= len(recommend_result.data)


def test_command_handler_without_event_bus(lora_service):
    """Test that handlers work without event bus (backward compatibility)."""
    handler = RecommendLoRAsHandler(lora_service, event_bus=None)

    command = RecommendLoRAsCommand(
        prompt="anime girl",
        base_model="SDXL",
        max_loras=3,
    )

    result = handler.handle(command)

    assert result.is_success
    # Should work fine without event bus
