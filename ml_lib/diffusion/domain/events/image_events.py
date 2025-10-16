"""Image generation domain events.

Events that occur during image generation lifecycle.
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from ml_lib.diffusion.domain.events.base import IDomainEvent


@dataclass(frozen=True)
class ImageGenerationRequestedEvent(IDomainEvent):
    """
    Event emitted when image generation is requested.

    Use cases:
    - Request tracking and analytics
    - Queue management
    - User activity monitoring

    Example:
        event = ImageGenerationRequestedEvent.create(
            prompt="anime girl with blue hair",
            negative_prompt="bad quality",
            base_model="SDXL",
            loras_requested=["style-lora-v1", "quality-lora-v2"],
        )
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=datetime.now)

    # Domain data
    prompt: str
    negative_prompt: str
    base_model: str
    loras_requested: list[str]
    seed: int | None

    @classmethod
    def create(
        cls,
        prompt: str,
        negative_prompt: str,
        base_model: str,
        loras_requested: list[str],
        seed: int | None = None,
    ) -> "ImageGenerationRequestedEvent":
        """Create event with auto-generated metadata."""
        return cls(
            event_id=str(uuid4()),
            occurred_at=datetime.now(),
            prompt=prompt,
            negative_prompt=negative_prompt,
            base_model=base_model,
            loras_requested=loras_requested,
            seed=seed,
        )


@dataclass(frozen=True)
class ImageGeneratedEvent(IDomainEvent):
    """
    Event emitted when image generation completes successfully.

    Use cases:
    - Success metrics and monitoring
    - Image cataloging
    - User notifications
    - Training data collection

    Example:
        event = ImageGeneratedEvent.create(
            image_path="/output/image_123.png",
            prompt="anime girl",
            base_model="SDXL",
            loras_used=["style-lora-v1"],
            generation_time_seconds=12.5,
            seed=42,
        )
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=datetime.now)

    # Domain data
    image_path: str
    prompt: str
    base_model: str
    loras_used: list[str]
    generation_time_seconds: float
    seed: int

    @classmethod
    def create(
        cls,
        image_path: str,
        prompt: str,
        base_model: str,
        loras_used: list[str],
        generation_time_seconds: float,
        seed: int,
    ) -> "ImageGeneratedEvent":
        """Create event with auto-generated metadata."""
        return cls(
            event_id=str(uuid4()),
            occurred_at=datetime.now(),
            image_path=image_path,
            prompt=prompt,
            base_model=base_model,
            loras_used=loras_used,
            generation_time_seconds=generation_time_seconds,
            seed=seed,
        )


@dataclass(frozen=True)
class ImageGenerationFailedEvent(IDomainEvent):
    """
    Event emitted when image generation fails.

    Use cases:
    - Error tracking and monitoring
    - Failure pattern analysis
    - User notifications
    - Retry logic

    Example:
        event = ImageGenerationFailedEvent.create(
            prompt="anime girl",
            base_model="SDXL",
            error_message="Out of memory",
            error_type="RuntimeError",
        )
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=datetime.now)

    # Domain data
    prompt: str
    base_model: str
    error_message: str
    error_type: str
    loras_attempted: list[str]

    @classmethod
    def create(
        cls,
        prompt: str,
        base_model: str,
        error_message: str,
        error_type: str,
        loras_attempted: list[str] | None = None,
    ) -> "ImageGenerationFailedEvent":
        """Create event with auto-generated metadata."""
        return cls(
            event_id=str(uuid4()),
            occurred_at=datetime.now(),
            prompt=prompt,
            base_model=base_model,
            error_message=error_message,
            error_type=error_type,
            loras_attempted=loras_attempted or [],
        )


@dataclass(frozen=True)
class PromptAnalyzedEvent(IDomainEvent):
    """
    Event emitted when a prompt is analyzed.

    Use cases:
    - Prompt optimization tracking
    - Intent analysis metrics
    - Tokenization pattern analysis

    Example:
        event = PromptAnalyzedEvent.create(
            original_prompt="girl with blue hair",
            optimized_prompt="anime girl, blue hair, detailed",
            detected_intent="character_portrait",
            complexity_score=0.6,
        )
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=datetime.now)

    # Domain data
    original_prompt: str
    optimized_prompt: str
    detected_intent: str
    complexity_score: float
    token_count: int

    @classmethod
    def create(
        cls,
        original_prompt: str,
        optimized_prompt: str,
        detected_intent: str,
        complexity_score: float,
        token_count: int,
    ) -> "PromptAnalyzedEvent":
        """Create event with auto-generated metadata."""
        return cls(
            event_id=str(uuid4()),
            occurred_at=datetime.now(),
            original_prompt=original_prompt,
            optimized_prompt=optimized_prompt,
            detected_intent=detected_intent,
            complexity_score=complexity_score,
            token_count=token_count,
        )
