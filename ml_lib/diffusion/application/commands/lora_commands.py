"""LoRA-related commands and handlers.

Commands for LoRA recommendation use cases.
"""

from dataclasses import dataclass
import asyncio
from typing import Optional
from ml_lib.diffusion.application.commands.base import (
    ICommand,
    ICommandHandler,
    CommandResult,
)
from ml_lib.diffusion.domain.services.lora_recommendation_service import (
    LoRARecommendationService,
)
from ml_lib.diffusion.domain.entities.lora import LoRARecommendation
from ml_lib.diffusion.domain.events.base import IEventBus
from ml_lib.diffusion.domain.events.lora_events import (
    LoRAsRecommendedEvent,
    TopLoRARecommendedEvent,
    LoRAFilteredEvent,
)


@dataclass(frozen=True)
class RecommendLoRAsCommand(ICommand):
    """
    Command to recommend LoRAs for a prompt.

    Use case: User wants LoRA recommendations for their image generation.
    """

    prompt: str
    base_model: str
    max_loras: int = 3
    min_confidence: float = 0.5


@dataclass(frozen=True)
class RecommendTopLoRACommand(ICommand):
    """
    Command to get single best LoRA recommendation.

    Use case: User wants only the best matching LoRA.
    """

    prompt: str
    base_model: str


@dataclass(frozen=True)
class FilterConfidentRecommendationsCommand(ICommand):
    """
    Command to filter recommendations to only confident ones.

    Use case: User wants to see only high-confidence recommendations.
    """

    recommendations: list[LoRARecommendation]


class RecommendLoRAsHandler(ICommandHandler[RecommendLoRAsCommand]):
    """
    Handler for RecommendLoRAsCommand.

    Coordinates with domain service to recommend LoRAs.
    Publishes LoRAsRecommendedEvent when successful.
    """

    def __init__(
        self,
        service: LoRARecommendationService,
        event_bus: Optional[IEventBus] = None
    ):
        """
        Initialize handler.

        Args:
            service: LoRA recommendation domain service
            event_bus: Optional event bus for publishing events
        """
        self.service = service
        self.event_bus = event_bus

    def handle(self, command: RecommendLoRAsCommand) -> CommandResult:
        """
        Handle LoRA recommendation.

        Args:
            command: RecommendLoRAsCommand with parameters

        Returns:
            CommandResult with list of LoRARecommendation
        """
        try:
            # Validate
            if not command.prompt:
                return CommandResult.validation_error("Prompt cannot be empty")

            if not command.base_model:
                return CommandResult.validation_error("Base model cannot be empty")

            if command.max_loras < 1:
                return CommandResult.validation_error(
                    "max_loras must be at least 1"
                )

            if not (0.0 <= command.min_confidence <= 1.0):
                return CommandResult.validation_error(
                    "min_confidence must be between 0 and 1"
                )

            # Execute
            recommendations = self.service.recommend(
                prompt=command.prompt,
                base_model=command.base_model,
                max_loras=command.max_loras,
                min_confidence=command.min_confidence,
            )

            # Publish event (if event bus available)
            if self.event_bus is not None:
                event = LoRAsRecommendedEvent.create(
                    prompt=command.prompt,
                    base_model=command.base_model,
                    lora_ids=[rec.lora.id for rec in recommendations],
                    recommendation_count=len(recommendations),
                    confidence_threshold=command.min_confidence,
                )
                # Fire and forget (async event publishing)
                asyncio.create_task(self.event_bus.publish(event))

            # Return result
            return CommandResult.success(
                data=recommendations,
                metadata={
                    "count": len(recommendations),
                    "prompt": command.prompt,
                    "base_model": command.base_model,
                },
            )

        except Exception as e:
            return CommandResult.failure(f"Failed to recommend LoRAs: {str(e)}")


class RecommendTopLoRAHandler(ICommandHandler[RecommendTopLoRACommand]):
    """
    Handler for RecommendTopLoRACommand.

    Gets single best LoRA recommendation.
    Publishes TopLoRARecommendedEvent when successful.
    """

    def __init__(
        self,
        service: LoRARecommendationService,
        event_bus: Optional[IEventBus] = None
    ):
        """
        Initialize handler.

        Args:
            service: LoRA recommendation domain service
            event_bus: Optional event bus for publishing events
        """
        self.service = service
        self.event_bus = event_bus

    def handle(self, command: RecommendTopLoRACommand) -> CommandResult:
        """
        Handle top LoRA recommendation.

        Args:
            command: RecommendTopLoRACommand with parameters

        Returns:
            CommandResult with single LoRARecommendation or None
        """
        try:
            # Validate
            if not command.prompt:
                return CommandResult.validation_error("Prompt cannot be empty")

            if not command.base_model:
                return CommandResult.validation_error("Base model cannot be empty")

            # Execute
            recommendation = self.service.recommend_top(
                prompt=command.prompt,
                base_model=command.base_model,
            )

            if recommendation is None:
                return CommandResult.not_found(
                    "No suitable LoRA found for this prompt and base model"
                )

            # Publish event (if event bus available)
            if self.event_bus is not None:
                event = TopLoRARecommendedEvent.create(
                    prompt=command.prompt,
                    base_model=command.base_model,
                    lora_id=recommendation.lora.id,
                    confidence=recommendation.confidence,
                )
                # Fire and forget (async event publishing)
                asyncio.create_task(self.event_bus.publish(event))

            # Return result
            return CommandResult.success(
                data=recommendation,
                metadata={
                    "prompt": command.prompt,
                    "base_model": command.base_model,
                    "lora_name": recommendation.lora.name.value,
                },
            )

        except Exception as e:
            return CommandResult.failure(f"Failed to recommend top LoRA: {str(e)}")


class FilterConfidentRecommendationsHandler(
    ICommandHandler[FilterConfidentRecommendationsCommand]
):
    """
    Handler for FilterConfidentRecommendationsCommand.

    Filters recommendations to only confident ones.
    Publishes LoRAFilteredEvent when successful.
    """

    def __init__(
        self,
        service: LoRARecommendationService,
        event_bus: Optional[IEventBus] = None
    ):
        """
        Initialize handler.

        Args:
            service: LoRA recommendation domain service
            event_bus: Optional event bus for publishing events
        """
        self.service = service
        self.event_bus = event_bus

    def handle(
        self, command: FilterConfidentRecommendationsCommand
    ) -> CommandResult:
        """
        Handle filtering confident recommendations.

        Args:
            command: FilterConfidentRecommendationsCommand with recommendations

        Returns:
            CommandResult with filtered list
        """
        try:
            # Validate
            if command.recommendations is None:
                return CommandResult.validation_error("Recommendations cannot be None")

            # Execute
            confident = self.service.filter_confident_recommendations(
                command.recommendations
            )

            # Publish event (if event bus available)
            if self.event_bus is not None and len(command.recommendations) > 0:
                # Get base model from first recommendation
                base_model = command.recommendations[0].lora.base_model.value
                event = LoRAFilteredEvent.create(
                    original_count=len(command.recommendations),
                    filtered_count=len(confident),
                    confidence_threshold=self.service.CONFIDENCE_THRESHOLD,
                    base_model=base_model,
                )
                # Fire and forget (async event publishing)
                asyncio.create_task(self.event_bus.publish(event))

            # Return result
            return CommandResult.success(
                data=confident,
                metadata={
                    "original_count": len(command.recommendations),
                    "filtered_count": len(confident),
                },
            )

        except Exception as e:
            return CommandResult.failure(
                f"Failed to filter recommendations: {str(e)}"
            )
