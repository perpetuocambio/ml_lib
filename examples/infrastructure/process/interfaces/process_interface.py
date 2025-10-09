"""Process execution interface."""

from abc import ABC, abstractmethod

from infrastructure.process.entities.command_arguments import CommandArguments
from infrastructure.process.entities.process_result import ProcessResult


class ProcessInterface(ABC):
    """Interface for executing system processes."""

    @abstractmethod
    def run(
        self, command: CommandArguments, timeout: int | None = None
    ) -> ProcessResult:
        """Execute a command and return the result."""
        pass
