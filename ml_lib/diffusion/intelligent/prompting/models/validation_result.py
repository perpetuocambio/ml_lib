"""Validation result model - replaces Dict[str, Any] return type."""

from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Result of character attribute validation.

    This class replaces the untyped Dict[str, Any] that was previously returned
    by validate_character_selection().

    All fields are strongly typed and documented.
    """

    is_valid: bool
    """Overall validation status - True if all checks passed."""

    compatibility_valid: bool
    """Whether selected attributes are compatible with each other."""

    issues: list[str] = field(default_factory=list)
    """Complete list of all validation issues found."""

    age_consistency_issues: list[str] = field(default_factory=list)
    """Issues related to age consistency (e.g., age-inappropriate attributes)."""

    blocked_content_issues: list[str] = field(default_factory=list)
    """Issues related to blocked/inappropriate content."""

    suggestions: list[str] = field(default_factory=list)
    """Suggestions for resolving validation issues."""

    @property
    def total_issue_count(self) -> int:
        """Total number of issues found."""
        return len(self.issues)

    @property
    def has_blocking_issues(self) -> bool:
        """Whether there are any blocking issues that prevent generation."""
        return len(self.blocked_content_issues) > 0

    @property
    def has_warnings(self) -> bool:
        """Whether there are non-blocking warnings."""
        return not self.is_valid and not self.has_blocking_issues
