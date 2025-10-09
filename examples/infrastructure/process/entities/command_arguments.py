"""Command arguments for process execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandArguments:
    """Type-safe command arguments."""

    arguments: list[str]

    def __post_init__(self) -> None:
        """Validate command arguments."""
        if not all(isinstance(arg, str) for arg in self.arguments):
            raise ValueError("All arguments must be strings")

    @classmethod
    def create(cls, arguments: list[str]) -> CommandArguments:
        """Create from list of arguments."""
        return cls(arguments=arguments)

    def to_list(self) -> list[str]:
        """Convert to list for compatibility."""
        return list(self.arguments)

    def is_empty(self) -> bool:
        """Check if empty."""
        return len(self.arguments) == 0
