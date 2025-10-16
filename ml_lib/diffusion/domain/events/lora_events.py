"""LoRA-related domain events.

Events that occur during LoRA recommendation and processing.
These are immutable records of things that happened in the domain.
"""

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4
from ml_lib.diffusion.domain.events.base import IDomainEvent
from ml_lib.diffusion.domain.entities.lora import LoRA


@dataclass(frozen=True)
class LoRAsRecommendedEvent(IDomainEvent):
    """
    Event emitted when LoRAs are recommended for a prompt.

    Use cases:
    - Logging recommendation history
    - Analytics on recommendation patterns
    - Caching popular recommendations
    - Training feedback loops

    Example:
        event = LoRAsRecommendedEvent.create(
            prompt="anime girl with blue hair",
            base_model="SDXL",
            lora_ids=["lora-123", "lora-456"],
            recommendation_count=3,
            confidence_threshold=0.7,
        )
        await event_bus.publish(event)
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=datetime.now)

    # Domain data
    prompt: str
    base_model: str
    lora_ids: list[str]
    recommendation_count: int
    confidence_threshold: float

    @classmethod
    def create(
        cls,
        prompt: str,
        base_model: str,
        lora_ids: list[str],
        recommendation_count: int,
        confidence_threshold: float = 0.0,
    ) -> "LoRAsRecommendedEvent":
        """
        Create event with auto-generated metadata.

        Args:
            prompt: User prompt
            base_model: Base model used
            lora_ids: IDs of recommended LoRAs
            recommendation_count: Number of recommendations
            confidence_threshold: Confidence threshold used

        Returns:
            New event instance
        """
        return cls(
            event_id=str(uuid4()),
            occurred_at=datetime.now(),
            prompt=prompt,
            base_model=base_model,
            lora_ids=lora_ids,
            recommendation_count=recommendation_count,
            confidence_threshold=confidence_threshold,
        )


@dataclass(frozen=True)
class TopLoRARecommendedEvent(IDomainEvent):
    """
    Event emitted when single best LoRA is recommended.

    Use cases:
    - Tracking most popular LoRAs
    - Single-LoRA optimization metrics
    - User preference learning

    Example:
        event = TopLoRARecommendedEvent.create(
            prompt="cyberpunk city",
            base_model="SDXL",
            lora_id="cyberpunk-lora-v2",
            confidence=0.95,
        )
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=datetime.now)

    # Domain data
    prompt: str
    base_model: str
    lora_id: str
    confidence: float

    @classmethod
    def create(
        cls,
        prompt: str,
        base_model: str,
        lora_id: str,
        confidence: float,
    ) -> "TopLoRARecommendedEvent":
        """Create event with auto-generated metadata."""
        return cls(
            event_id=str(uuid4()),
            occurred_at=datetime.now(),
            prompt=prompt,
            base_model=base_model,
            lora_id=lora_id,
            confidence=confidence,
        )


@dataclass(frozen=True)
class LoRALoadedEvent(IDomainEvent):
    """
    Event emitted when a LoRA model is loaded from repository.

    Use cases:
    - Caching frequently loaded LoRAs
    - Performance monitoring
    - Repository access patterns

    Example:
        event = LoRALoadedEvent.create(
            lora_id="anime-style-v1",
            lora_name="Anime Art Style V1",
            base_model="SDXL",
        )
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=datetime.now)

    # Domain data
    lora_id: str
    lora_name: str
    base_model: str

    @classmethod
    def create(
        cls,
        lora_id: str,
        lora_name: str,
        base_model: str,
    ) -> "LoRALoadedEvent":
        """Create event with auto-generated metadata."""
        return cls(
            event_id=str(uuid4()),
            occurred_at=datetime.now(),
            lora_id=lora_id,
            lora_name=lora_name,
            base_model=base_model,
        )


@dataclass(frozen=True)
class LoRAFilteredEvent(IDomainEvent):
    """
    Event emitted when LoRAs are filtered by confidence threshold.

    Use cases:
    - Monitoring filter effectiveness
    - Optimizing confidence thresholds
    - Quality metrics

    Example:
        event = LoRAFilteredEvent.create(
            original_count=10,
            filtered_count=3,
            confidence_threshold=0.7,
            base_model="SDXL",
        )
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=datetime.now)

    # Domain data
    original_count: int
    filtered_count: int
    confidence_threshold: float
    base_model: str

    @classmethod
    def create(
        cls,
        original_count: int,
        filtered_count: int,
        confidence_threshold: float,
        base_model: str,
    ) -> "LoRAFilteredEvent":
        """Create event with auto-generated metadata."""
        return cls(
            event_id=str(uuid4()),
            occurred_at=datetime.now(),
            original_count=original_count,
            filtered_count=filtered_count,
            confidence_threshold=confidence_threshold,
            base_model=base_model,
        )
