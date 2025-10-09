import platform
import subprocess

from infrastructure.process.entities.command_arguments import CommandArguments
from infrastructure.process.entities.process_result import ProcessResult
from infrastructure.process.interfaces.process_interface import ProcessInterface


class SubprocessService(ProcessInterface):
    """Concrete implementation of ProcessInterface using Python's subprocess module."""

    def run(
        self, command: CommandArguments, timeout: int | None = None
    ) -> ProcessResult:
        """Execute a command using subprocess and return the result."""
        completed = subprocess.run(
            command.to_list(),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return ProcessResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )

    @property
    def is_windows(self) -> bool:
        """Check if the current platform is Windows."""
        return platform.system().lower() == "windows"

    @property
    def is_linux(self) -> bool:
        """Check if the current platform is Linux."""
        return platform.system().lower() == "linux"
