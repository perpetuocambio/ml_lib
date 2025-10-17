"""
Safety check result value objects.

This module provides type-safe safety check results,
replacing tuple returns.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptBlockResult:
    """Result of checking if a prompt should be blocked."""

    should_block: bool
    safety_score: float  # 0.0 (unsafe) to 1.0 (safe)
    violations: list[str]

    def __post_init__(self) -> None:
        """Validate result."""
        if not 0.0 <= self.safety_score <= 1.0:
            raise ValueError(f"safety_score must be 0.0-1.0, got {self.safety_score}")


@dataclass(frozen=True)
class PromptSafetyResult:
    """Result of checking if a prompt is safe."""

    is_safe: bool
    violations: list[str]

    def has_violations(self) -> bool:
        """Check if there are any violations."""
        return len(self.violations) > 0

    def violation_count(self) -> int:
        """Get number of violations."""
        return len(self.violations)
