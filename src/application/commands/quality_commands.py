"""Quality validation commands and handlers.

Commands for validating image generation quality.
"""

from dataclasses import dataclass
from pathlib import Path
from ml_lib.diffusion.application.commands.base import (
    ICommand,
    ICommandHandler,
    CommandResult,
)
from ml_lib.diffusion.domain.services.quality_validation_service import (
    QualityValidationService,
)


@dataclass(frozen=True)
class ValidateImageQualityCommand(ICommand):
    """
    Command to validate a single generated image.

    Use case: After image generation, validate quality before delivery.
    """

    image_path: Path
    expected_width: int = 1024
    expected_height: int = 1024
    min_quality_threshold: float = 70.0


@dataclass(frozen=True)
class ValidateBatchQualityCommand(ICommand):
    """
    Command to validate multiple generated images.

    Use case: Batch validation of multiple images.
    """

    image_paths: list[Path]
    expected_width: int = 1024
    expected_height: int = 1024
    min_quality_threshold: float = 70.0


class ValidateImageQualityHandler(ICommandHandler[ValidateImageQualityCommand]):
    """
    Handler for ValidateImageQualityCommand.

    Validates single image quality using domain service.
    """

    def __init__(self, validation_service: QualityValidationService):
        """
        Initialize handler.

        Args:
            validation_service: Quality validation domain service
        """
        self.validation_service = validation_service

    def handle(self, command: ValidateImageQualityCommand) -> CommandResult:
        """
        Handle image quality validation.

        Args:
            command: ValidateImageQualityCommand with parameters

        Returns:
            CommandResult with QualityMetrics
        """
        try:
            # Validate
            if not command.image_path:
                return CommandResult.validation_error("Image path cannot be empty")

            if command.expected_width < 1 or command.expected_height < 1:
                return CommandResult.validation_error(
                    "Expected dimensions must be positive"
                )

            if not (0.0 <= command.min_quality_threshold <= 100.0):
                return CommandResult.validation_error(
                    "Quality threshold must be between 0 and 100"
                )

            # Execute validation
            metrics = self.validation_service.validate_image(
                image_path=command.image_path,
                expected_width=command.expected_width,
                expected_height=command.expected_height,
            )

            # Check if meets threshold
            if metrics.quality_score < command.min_quality_threshold:
                return CommandResult.failure(
                    f"Image quality {metrics.quality_score:.0f}/100 "
                    f"below threshold {command.min_quality_threshold:.0f}/100",
                )

            # Return success with metrics
            return CommandResult.success(
                data=metrics,
                metadata={
                    "quality_score": metrics.quality_score,
                    "is_valid": metrics.is_valid,
                    "is_excellent": metrics.is_excellent,
                    "resolution": f"{metrics.actual_width}x{metrics.actual_height}",
                },
            )

        except Exception as e:
            return CommandResult.failure(f"Failed to validate image: {str(e)}")


class ValidateBatchQualityHandler(ICommandHandler[ValidateBatchQualityCommand]):
    """
    Handler for ValidateBatchQualityCommand.

    Validates multiple images in batch.
    """

    def __init__(self, validation_service: QualityValidationService):
        """
        Initialize handler.

        Args:
            validation_service: Quality validation domain service
        """
        self.validation_service = validation_service

    def handle(self, command: ValidateBatchQualityCommand) -> CommandResult:
        """
        Handle batch quality validation.

        Args:
            command: ValidateBatchQualityCommand with parameters

        Returns:
            CommandResult with list of QualityMetrics and summary stats
        """
        try:
            # Validate
            if not command.image_paths:
                return CommandResult.validation_error("Image paths list cannot be empty")

            if command.expected_width < 1 or command.expected_height < 1:
                return CommandResult.validation_error(
                    "Expected dimensions must be positive"
                )

            # Execute validation
            metrics_list = self.validation_service.validate_batch(
                image_paths=command.image_paths,
                expected_width=command.expected_width,
                expected_height=command.expected_height,
            )

            # Calculate summary statistics
            summary = self.validation_service.get_summary_stats(metrics_list)

            # Check if batch meets threshold
            if summary["pass_rate"] < command.min_quality_threshold:
                return CommandResult.failure(
                    f"Batch pass rate {summary['pass_rate']:.1f}% "
                    f"below threshold {command.min_quality_threshold:.0f}%"
                )

            # Return success with all metrics
            return CommandResult.success(
                data=metrics_list,
                metadata={
                    "total_images": summary["total"],
                    "passed": summary["passed"],
                    "failed": summary["failed"],
                    "pass_rate": summary["pass_rate"],
                    "avg_quality": summary["avg_quality"],
                    "min_quality": summary["min_quality"],
                    "max_quality": summary["max_quality"],
                },
            )

        except Exception as e:
            return CommandResult.failure(f"Failed to validate batch: {str(e)}")
