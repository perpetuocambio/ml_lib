"""Example event handlers.

Demonstrates Observer pattern with concrete handler implementations.
These handlers react to domain events and trigger side effects.
"""

import logging
from ml_lib.diffusion.domain.events.base import IEventHandler
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

logger = logging.getLogger(__name__)


class LoggingEventHandler(IEventHandler[LoRAsRecommendedEvent]):
    """
    Example handler that logs LoRA recommendations.

    Use cases:
    - Audit trail
    - Debugging
    - Recommendation history

    Example:
        handler = LoggingEventHandler()
        bus.subscribe(LoRAsRecommendedEvent, handler)
    """

    async def handle(self, event: LoRAsRecommendedEvent) -> None:
        """
        Log LoRA recommendation event.

        Args:
            event: LoRAsRecommendedEvent
        """
        logger.info(
            f"LoRAs recommended: {event.recommendation_count} LoRAs "
            f"for prompt '{event.prompt}' (model: {event.base_model})"
        )
        logger.debug(f"Recommended LoRA IDs: {event.lora_ids}")


class MetricsEventHandler(IEventHandler[ImageGeneratedEvent]):
    """
    Example handler that tracks image generation metrics.

    Use cases:
    - Performance monitoring
    - Usage analytics
    - Optimization insights

    This is a simplified example. In production, this would:
    - Send metrics to monitoring system (Prometheus, DataDog, etc.)
    - Update dashboards
    - Track SLAs

    Example:
        handler = MetricsEventHandler()
        bus.subscribe(ImageGeneratedEvent, handler)
    """

    def __init__(self):
        """Initialize metrics handler."""
        self.total_images_generated = 0
        self.total_generation_time = 0.0
        self.images_by_model: dict[str, int] = {}

    async def handle(self, event: ImageGeneratedEvent) -> None:
        """
        Track image generation metrics.

        Args:
            event: ImageGeneratedEvent
        """
        self.total_images_generated += 1
        self.total_generation_time += event.generation_time_seconds

        # Track by model
        model = event.base_model
        self.images_by_model[model] = self.images_by_model.get(model, 0) + 1

        # Log metrics
        avg_time = self.total_generation_time / self.total_images_generated
        logger.info(
            f"Image generated in {event.generation_time_seconds:.2f}s "
            f"(avg: {avg_time:.2f}s, total: {self.total_images_generated})"
        )

    def get_average_generation_time(self) -> float:
        """Get average generation time across all images."""
        if self.total_images_generated == 0:
            return 0.0
        return self.total_generation_time / self.total_images_generated


class ErrorLoggingHandler(IEventHandler[ImageGenerationFailedEvent]):
    """
    Example handler that logs and tracks image generation failures.

    Use cases:
    - Error monitoring
    - Failure pattern detection
    - Alerting

    Example:
        handler = ErrorLoggingHandler()
        bus.subscribe(ImageGenerationFailedEvent, handler)
    """

    def __init__(self):
        """Initialize error logging handler."""
        self.error_count = 0
        self.errors_by_type: dict[str, int] = {}

    async def handle(self, event: ImageGenerationFailedEvent) -> None:
        """
        Log image generation failure.

        Args:
            event: ImageGenerationFailedEvent
        """
        self.error_count += 1

        # Track by error type
        error_type = event.error_type
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1

        # Log error
        logger.error(
            f"Image generation failed: {event.error_message} "
            f"(type: {error_type}, prompt: '{event.prompt}')"
        )
        logger.debug(f"LoRAs attempted: {event.loras_attempted}")

        # Alert if error rate is high (example threshold)
        if self.error_count > 10:
            logger.warning(
                f"High error rate detected: {self.error_count} failures"
            )


class CachingHandler(IEventHandler[LoRALoadedEvent]):
    """
    Example handler that tracks frequently loaded LoRAs for caching.

    Use cases:
    - Cache warming
    - Performance optimization
    - Usage pattern analysis

    Example:
        handler = CachingHandler()
        bus.subscribe(LoRALoadedEvent, handler)
    """

    def __init__(self):
        """Initialize caching handler."""
        self.load_counts: dict[str, int] = {}

    async def handle(self, event: LoRALoadedEvent) -> None:
        """
        Track LoRA load event.

        Args:
            event: LoRALoadedEvent
        """
        lora_id = event.lora_id
        self.load_counts[lora_id] = self.load_counts.get(lora_id, 0) + 1

        load_count = self.load_counts[lora_id]
        logger.debug(
            f"LoRA '{event.lora_name}' loaded (total loads: {load_count})"
        )

        # Suggest caching if frequently loaded
        if load_count >= 5:
            logger.info(
                f"LoRA '{event.lora_name}' loaded {load_count} times - "
                f"consider caching"
            )

    def get_most_loaded_loras(self, top_n: int = 5) -> list[tuple[str, int]]:
        """
        Get most frequently loaded LoRAs.

        Args:
            top_n: Number of top LoRAs to return

        Returns:
            List of (lora_id, load_count) tuples
        """
        sorted_loras = sorted(
            self.load_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_loras[:top_n]


class PromptAnalyticsHandler(IEventHandler[PromptAnalyzedEvent]):
    """
    Example handler that analyzes prompt patterns.

    Use cases:
    - Prompt optimization insights
    - User behavior analysis
    - Training data for improvements

    Example:
        handler = PromptAnalyticsHandler()
        bus.subscribe(PromptAnalyzedEvent, handler)
    """

    def __init__(self):
        """Initialize prompt analytics handler."""
        self.prompt_count = 0
        self.total_complexity = 0.0
        self.intents: dict[str, int] = {}

    async def handle(self, event: PromptAnalyzedEvent) -> None:
        """
        Analyze prompt event.

        Args:
            event: PromptAnalyzedEvent
        """
        self.prompt_count += 1
        self.total_complexity += event.complexity_score

        # Track intent distribution
        intent = event.detected_intent
        self.intents[intent] = self.intents.get(intent, 0) + 1

        logger.debug(
            f"Prompt analyzed: '{event.original_prompt}' -> "
            f"'{event.optimized_prompt}' (intent: {intent}, "
            f"complexity: {event.complexity_score:.2f})"
        )

    def get_average_complexity(self) -> float:
        """Get average prompt complexity score."""
        if self.prompt_count == 0:
            return 0.0
        return self.total_complexity / self.prompt_count

    def get_intent_distribution(self) -> dict[str, float]:
        """
        Get distribution of detected intents.

        Returns:
            Dictionary mapping intent to percentage
        """
        if self.prompt_count == 0:
            return {}

        return {
            intent: (count / self.prompt_count) * 100
            for intent, count in self.intents.items()
        }


class MultiEventHandler(
    IEventHandler[ImageGeneratedEvent],
    IEventHandler[ImageGenerationFailedEvent],
):
    """
    Example handler that subscribes to multiple event types.

    Demonstrates that a single handler can implement multiple IEventHandler interfaces.

    Use cases:
    - Consolidated monitoring
    - Unified logging
    - State tracking across events

    Example:
        handler = MultiEventHandler()
        bus.subscribe(ImageGeneratedEvent, handler)
        bus.subscribe(ImageGenerationFailedEvent, handler)
    """

    def __init__(self):
        """Initialize multi-event handler."""
        self.success_count = 0
        self.failure_count = 0

    async def handle(
        self,
        event: ImageGeneratedEvent | ImageGenerationFailedEvent
    ) -> None:
        """
        Handle both success and failure events.

        Args:
            event: ImageGeneratedEvent or ImageGenerationFailedEvent
        """
        if isinstance(event, ImageGeneratedEvent):
            self.success_count += 1
            logger.info(f"Image generation succeeded (total: {self.success_count})")
        elif isinstance(event, ImageGenerationFailedEvent):
            self.failure_count += 1
            logger.warning(f"Image generation failed (total: {self.failure_count})")

    def get_success_rate(self) -> float:
        """
        Calculate success rate.

        Returns:
            Success rate as percentage (0-100)
        """
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return (self.success_count / total) * 100
