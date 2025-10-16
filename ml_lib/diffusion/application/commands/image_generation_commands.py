"""Image Generation Commands.

Commands for image generation use cases, wrapping the GenerateImageUseCase.
"""

from dataclasses import dataclass
from typing import Optional
from ml_lib.diffusion.application.commands.base import (
    ICommand,
    ICommandHandler,
    CommandResult,
)
from ml_lib.diffusion.application.use_cases.generate_image import (
    GenerateImageUseCase,
    GenerateImageRequest,
)


@dataclass(frozen=True)
class GenerateImageCommand(ICommand):
    """
    Command to generate an image.

    This wraps the GenerateImageUseCase, allowing it to be dispatched
    through the CommandBus with all benefits of the Command Pattern:
    - Logging
    - Queuing (future)
    - Transaction management (future)
    - Retry logic (future)

    Use case: User wants to generate an image from a prompt.
    """

    prompt: str
    negative_prompt: str = ""
    base_model: str = "SDXL"
    seed: Optional[int] = None
    num_steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    max_loras: int = 3
    min_lora_confidence: float = 0.5


class GenerateImageHandler(ICommandHandler[GenerateImageCommand]):
    """
    Handler for GenerateImageCommand.

    Delegates to GenerateImageUseCase for orchestration.
    The handler's responsibility is:
    - Command validation
    - DTO conversion
    - Error handling
    - Result wrapping

    The use case's responsibility is:
    - Business flow orchestration
    - Service coordination
    """

    def __init__(self, use_case: GenerateImageUseCase):
        """
        Initialize handler.

        Args:
            use_case: Image generation use case
        """
        self.use_case = use_case

    def handle(self, command: GenerateImageCommand) -> CommandResult:
        """
        Handle image generation command.

        Args:
            command: GenerateImageCommand with parameters

        Returns:
            CommandResult with GenerateImageResult
        """
        try:
            # Validate
            if not command.prompt:
                return CommandResult.validation_error("Prompt cannot be empty")

            if not command.base_model:
                return CommandResult.validation_error("Base model cannot be empty")

            if command.max_loras < 0:
                return CommandResult.validation_error(
                    "max_loras must be non-negative"
                )

            if not (0.0 <= command.min_lora_confidence <= 1.0):
                return CommandResult.validation_error(
                    "min_lora_confidence must be between 0 and 1"
                )

            # Convert Command to Request DTO
            request = GenerateImageRequest(
                prompt=command.prompt,
                negative_prompt=command.negative_prompt,
                base_model=command.base_model,
                seed=command.seed,
                num_steps=command.num_steps,
                cfg_scale=command.cfg_scale,
                width=command.width,
                height=command.height,
                max_loras=command.max_loras,
                min_lora_confidence=command.min_lora_confidence,
            )

            # Execute use case
            result = self.use_case.execute(request)

            # Return success with result
            return CommandResult.success(
                data=result,
                metadata={
                    "prompt": command.prompt,
                    "base_model": command.base_model,
                    "seed": result.seed,
                    "loras_count": len(result.loras_applied),
                    "generation_time": result.generation_time_seconds,
                },
            )

        except ValueError as e:
            return CommandResult.validation_error(str(e))
        except Exception as e:
            return CommandResult.failure(f"Failed to generate image: {str(e)}")


@dataclass(frozen=True)
class QuickGenerateCommand(ICommand):
    """
    Command for quick image generation with minimal parameters.

    Convenience command for simple use cases where user just provides
    a prompt and wants sensible defaults.

    Use case: User wants quick generation without tweaking parameters.
    """

    prompt: str
    base_model: str = "SDXL"


class QuickGenerateHandler(ICommandHandler[QuickGenerateCommand]):
    """
    Handler for QuickGenerateCommand.

    Wraps GenerateImageHandler with simplified parameters.
    """

    def __init__(self, use_case: GenerateImageUseCase):
        """
        Initialize handler.

        Args:
            use_case: Image generation use case
        """
        self.use_case = use_case

    def handle(self, command: QuickGenerateCommand) -> CommandResult:
        """
        Handle quick generation command.

        Args:
            command: QuickGenerateCommand with minimal parameters

        Returns:
            CommandResult with GenerateImageResult
        """
        # Delegate to full GenerateImageHandler with defaults
        full_command = GenerateImageCommand(
            prompt=command.prompt,
            base_model=command.base_model,
            # Use all defaults for other parameters
        )

        handler = GenerateImageHandler(self.use_case)
        return handler.handle(full_command)
