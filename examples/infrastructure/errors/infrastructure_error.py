from datetime import datetime

from infrastructure.errors.error_context import ErrorContext


class InfrastructureError(Exception):
    """Base class for all infrastructure-related exceptions."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        context: ErrorContext | None = None,
        original_exception: Exception | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or ErrorContext.empty()
        self.original_exception = original_exception
        self.timestamp = datetime.now()

    def get_error_summary(self) -> str:
        """Get comprehensive error summary for logging/debugging."""
        summary_parts = [
            f"Error: {self.__class__.__name__}",
            f"Message: {self.message}",
            f"Timestamp: {self.timestamp.isoformat()}",
        ]

        if self.error_code:
            summary_parts.append(f"Code: {self.error_code}")

        if not self.context.is_empty():
            summary_parts.append(f"Context: {self.context.get_summary()}")

        if self.original_exception:
            summary_parts.append(
                f"Original: {type(self.original_exception).__name__}: {self.original_exception}"
            )

        return " | ".join(summary_parts)

    def is_retryable(self) -> bool:
        """Determine if this error represents a retryable condition."""
        # Base implementation - subclasses should override for specific logic
        return False

    def get_user_friendly_message(self) -> str:
        """Get a user-friendly error message (hiding technical details)."""
        return self.message

    def add_text_context(self, key: str, value: str) -> None:
        """Add text contextual information to the error."""
        self.context = self.context.add_text_entry(key, value)

    def add_numeric_context(self, key: str, value: int | float) -> None:
        """Add numeric contextual information to the error."""
        self.context = self.context.add_numeric_entry(key, value)

    def add_boolean_context(self, key: str, value: bool) -> None:
        """Add boolean contextual information to the error."""
        self.context = self.context.add_boolean_entry(key, value)

    def has_context(self, key: str) -> bool:
        """Check if specific context key exists."""
        return self.context.has_key(key)

    def get_text_context_value(self, key: str) -> str | None:
        """Get text value from error context."""
        return self.context.get_text_value(key)

    def get_numeric_context_value(self, key: str) -> int | float | None:
        """Get numeric value from error context."""
        return self.context.get_numeric_value(key)

    def get_boolean_context_value(self, key: str) -> bool | None:
        """Get boolean value from error context."""
        return self.context.get_boolean_value(key)
