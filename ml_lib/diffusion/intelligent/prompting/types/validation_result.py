"""Validation result type for character attribute validation."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ValidationResult:
    """Result of validating character attribute selections.

    Replaces Dict[str, Any] returns with strongly typed class.
    """

    is_valid: bool
    compatibility_valid: bool
    issues: List[str] = field(default_factory=list)
    age_consistency_issues: List[str] = field(default_factory=list)
    blocked_content_issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    @property
    def total_issue_count(self) -> int:
        """Total number of issues found."""
        return len(self.issues)

    @property
    def has_age_issues(self) -> bool:
        """Check if there are age consistency issues."""
        return len(self.age_consistency_issues) > 0

    @property
    def has_blocked_content(self) -> bool:
        """Check if there is blocked content."""
        return len(self.blocked_content_issues) > 0

    def __bool__(self) -> bool:
        """Boolean evaluation returns is_valid."""
        return self.is_valid
