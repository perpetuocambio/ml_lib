from dataclasses import dataclass, field

from ml_lib.content_safety.safe_status import SafetyStatus


@dataclass(frozen=True)
class SafetyCheckResult:
    """Result of a safety check.

    Attributes:
        status: Safety status (safe, unsafe, warning).
        _blocked_keywords: Internal list of blocked keywords found.
        _warnings: Internal list of warning messages.

    Example:
        >>> result = SafetyCheckResult(
        ...     status=SafetyStatus.SAFE,
        ...     _blocked_keywords=[],
        ...     _warnings=[]
        ... )
        >>> print(result.is_safe)
        True
    """

    status: SafetyStatus
    _blocked_keywords: list[str] = field(default_factory=list)
    _warnings: list[str] = field(default_factory=list)

    @property
    def is_safe(self) -> bool:
        """Check if content is safe."""
        return self.status == SafetyStatus.SAFE

    @property
    def is_unsafe(self) -> bool:
        """Check if content is unsafe."""
        return self.status == SafetyStatus.UNSAFE

    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return len(self._warnings) > 0

    @property
    def has_blocked_keywords(self) -> bool:
        """Check if blocked keywords were found."""
        return len(self._blocked_keywords) > 0

    def get_blocked_keywords(self) -> list[str]:
        """Get all blocked keywords.

        Returns:
            List of blocked keywords.
        """
        return self._blocked_keywords.copy()

    def get_warnings(self) -> list[str]:
        """Get all warnings.

        Returns:
            List of warning messages.
        """
        return self._warnings.copy()
