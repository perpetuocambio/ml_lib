from dataclasses import dataclass


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
