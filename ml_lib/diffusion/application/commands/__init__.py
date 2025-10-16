"""Application Commands - Command Pattern implementation.

Commands represent user intentions and business operations.
They encapsulate all data needed to perform an action.

Following CQRS principles:
- Commands: Write operations (mutations)
- Queries: Read operations (will be separate)

Command Pattern benefits:
- Decouples sender from receiver
- Commands are objects (can be logged, queued, undone)
- Easy to add new operations (Open/Closed Principle)
- Testable in isolation
"""

from ml_lib.diffusion.application.commands.base import (
    ICommand,
    ICommandHandler,
    ICommandBus,
    CommandResult,
    CommandStatus,
)
from ml_lib.diffusion.application.commands.bus import CommandBus
from ml_lib.diffusion.application.commands.lora_commands import (
    RecommendLoRAsCommand,
    RecommendTopLoRACommand,
    FilterConfidentRecommendationsCommand,
    RecommendLoRAsHandler,
    RecommendTopLoRAHandler,
    FilterConfidentRecommendationsHandler,
)
from ml_lib.diffusion.application.commands.image_generation_commands import (
    GenerateImageCommand,
    QuickGenerateCommand,
    GenerateImageHandler,
    QuickGenerateHandler,
)

__all__ = [
    # Base interfaces
    "ICommand",
    "ICommandHandler",
    "ICommandBus",
    "CommandResult",
    "CommandStatus",
    # Implementations
    "CommandBus",
    # LoRA commands
    "RecommendLoRAsCommand",
    "RecommendTopLoRACommand",
    "FilterConfidentRecommendationsCommand",
    # LoRA handlers
    "RecommendLoRAsHandler",
    "RecommendTopLoRAHandler",
    "FilterConfidentRecommendationsHandler",
    # Image generation commands
    "GenerateImageCommand",
    "QuickGenerateCommand",
    # Image generation handlers
    "GenerateImageHandler",
    "QuickGenerateHandler",
]
