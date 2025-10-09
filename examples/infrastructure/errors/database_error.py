import re

from infrastructure.errors.error_context import ErrorContext
from infrastructure.errors.infrastructure_error import InfrastructureError


class DatabaseError(InfrastructureError):
    """Raised for errors related to database operations."""

    def __init__(
        self,
        message: str,
        table_name: str | None = None,
        query: str | None = None,
        connection_string: str | None = None,
        row_count: int | None = None,
        transaction_id: str | None = None,
        error_code: str | None = None,
        original_exception: Exception | None = None,
    ):
        context = ErrorContext.empty()

        if table_name:
            context = context.add_text_entry("table_name", table_name)
        if query:
            # Truncate long queries for security and readability
            context = context.add_text_entry(
                "query", query[:500] if len(query) > 500 else query
            )
        if connection_string:
            # Remove sensitive information from connection string
            safe_connection = self._sanitize_connection_string(connection_string)
            context = context.add_text_entry("connection_string", safe_connection)
        if transaction_id:
            context = context.add_text_entry("transaction_id", transaction_id)
        if row_count is not None:
            context = context.add_numeric_entry("row_count", row_count)

        super().__init__(
            message=message,
            error_code=error_code or "DATABASE_ERROR",
            context=context,
            original_exception=original_exception,
        )

    def _sanitize_connection_string(self, connection_string: str) -> str:
        """Remove passwords and sensitive data from connection string."""
        # Remove password patterns

        patterns = [
            r"password=([^;]+)",
            r"pwd=([^;]+)",
            r"pass=([^;]+)",
        ]
        sanitized = connection_string
        for pattern in patterns:
            sanitized = re.sub(
                pattern,
                lambda m: f"{m.group(0).split('=')[0]}=***",
                sanitized,
                flags=re.IGNORECASE,
            )
        return sanitized

    def is_retryable(self) -> bool:
        """Determine if database error is retryable."""
        error_code = self.error_code
        if error_code:
            # Common retryable database errors
            retryable_codes = {
                "CONNECTION_TIMEOUT",
                "CONNECTION_LOST",
                "DEADLOCK",
                "LOCK_TIMEOUT",
                "TEMPORARY_FAILURE",
                "CONNECTION_REFUSED",
            }
            return error_code in retryable_codes
        return False

    def get_user_friendly_message(self) -> str:
        """Get user-friendly database error message."""
        table_name = self.get_text_context_value("table_name")
        transaction_id = self.get_text_context_value("transaction_id")

        if table_name and transaction_id:
            return f"Database error in table '{table_name}' (transaction: {transaction_id})"
        elif table_name:
            return f"Database error in table '{table_name}'"
        elif transaction_id:
            return f"Database error in transaction '{transaction_id}'"
        else:
            return f"Database operation failed: {self.message}"

    @classmethod
    def connection_failed(
        cls, connection_string: str, original_exception: Exception | None = None
    ) -> "DatabaseError":
        """Create error for database connection failure."""
        return cls(
            message="Failed to connect to database",
            connection_string=connection_string,
            error_code="CONNECTION_FAILED",
            original_exception=original_exception,
        )

    @classmethod
    def query_failed(
        cls,
        query: str,
        table_name: str | None = None,
        original_exception: Exception | None = None,
    ) -> "DatabaseError":
        """Create error for query execution failure."""
        message = "Database query failed"
        if table_name:
            message += f" on table '{table_name}'"
        return cls(
            message=message,
            table_name=table_name,
            query=query,
            error_code="QUERY_FAILED",
            original_exception=original_exception,
        )

    @classmethod
    def transaction_failed(
        cls, transaction_id: str, operation: str | None = None
    ) -> "DatabaseError":
        """Create error for transaction failure."""
        message = f"Database transaction '{transaction_id}' failed"
        if operation:
            message += f" during {operation}"
        return cls(
            message=message,
            transaction_id=transaction_id,
            error_code="TRANSACTION_FAILED",
        )

    @classmethod
    def constraint_violation(
        cls, table_name: str, constraint_type: str = "unknown"
    ) -> "DatabaseError":
        """Create error for database constraint violation."""
        return cls(
            message=f"Database constraint violation in table '{table_name}' ({constraint_type})",
            table_name=table_name,
            error_code="CONSTRAINT_VIOLATION",
        )
