"""Base Command interfaces and types.

Defines the core Command Pattern interfaces following CQRS principles.
"""

from typing import Protocol, TypeVar, Generic, runtime_checkable, Any
from dataclasses import dataclass
from enum import Enum, auto


class CommandStatus(Enum):
    """Status of command execution."""

    SUCCESS = auto()
    FAILED = auto()
    VALIDATION_ERROR = auto()
    NOT_FOUND = auto()
    CONFLICT = auto()


@dataclass(frozen=True)
class CommandResult:
    """Result of command execution.

    Attributes:
        status: Execution status
        data: Optional result data
        error: Optional error message
        metadata: Optional metadata (e.g., execution time, affected records)
    """

    status: CommandStatus
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] | None = None

    @property
    def is_success(self) -> bool:
        """Check if command succeeded."""
        return self.status == CommandStatus.SUCCESS

    @property
    def is_failure(self) -> bool:
        """Check if command failed."""
        return self.status != CommandStatus.SUCCESS

    @classmethod
    def success(cls, data: Any = None, metadata: dict | None = None) -> "CommandResult":
        """Create success result."""
        return cls(status=CommandStatus.SUCCESS, data=data, metadata=metadata)

    @classmethod
    def failure(
        cls, error: str, status: CommandStatus = CommandStatus.FAILED
    ) -> "CommandResult":
        """Create failure result."""
        return cls(status=status, error=error)

    @classmethod
    def validation_error(cls, error: str) -> "CommandResult":
        """Create validation error result."""
        return cls(status=CommandStatus.VALIDATION_ERROR, error=error)

    @classmethod
    def not_found(cls, error: str) -> "CommandResult":
        """Create not found result."""
        return cls(status=CommandStatus.NOT_FOUND, error=error)

    @classmethod
    def conflict(cls, error: str) -> "CommandResult":
        """Create conflict result."""
        return cls(status=CommandStatus.CONFLICT, error=error)


@runtime_checkable
class ICommand(Protocol):
    """
    Base interface for all commands.

    Commands are immutable data structures that represent user intentions.
    They should be frozen dataclasses with all required data for execution.

    Example:
        @dataclass(frozen=True)
        class RecommendLoRACommand(ICommand):
            prompt: str
            base_model: str
            max_loras: int = 3
    """

    pass


# Type variable for command generic typing
TCommand = TypeVar("TCommand", bound=ICommand, contravariant=True)


@runtime_checkable
class ICommandHandler(Protocol, Generic[TCommand]):
    """
    Handler for a specific command type.

    Each command should have exactly one handler (Single Responsibility).
    Handlers contain the business logic for executing commands.

    Example:
        class RecommendLoRAHandler(ICommandHandler[RecommendLoRACommand]):
            def __init__(self, service: LoRARecommendationService):
                self.service = service

            def handle(self, command: RecommendLoRACommand) -> CommandResult:
                # Execute business logic
                ...
    """

    def handle(self, command: TCommand) -> CommandResult:
        """
        Handle command execution.

        Args:
            command: Command to execute

        Returns:
            CommandResult with status and optional data/error
        """
        ...


@runtime_checkable
class ICommandBus(Protocol):
    """
    Command bus for dispatching commands to handlers.

    Provides centralized command execution with:
    - Handler registration
    - Command dispatching
    - Middleware support (logging, validation, transactions)
    - Error handling

    Example:
        bus = CommandBus()
        bus.register(RecommendLoRACommand, RecommendLoRAHandler(service))
        result = bus.dispatch(RecommendLoRACommand(prompt="...", base_model="..."))
    """

    def register(self, command_type: type[ICommand], handler: ICommandHandler) -> None:
        """
        Register handler for command type.

        Args:
            command_type: Command class
            handler: Handler instance
        """
        ...

    def dispatch(self, command: ICommand) -> CommandResult:
        """
        Dispatch command to appropriate handler.

        Args:
            command: Command to execute

        Returns:
            CommandResult from handler

        Raises:
            ValueError: If no handler registered for command type
        """
        ...
