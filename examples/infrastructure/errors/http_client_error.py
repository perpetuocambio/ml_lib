from infrastructure.errors.error_context import ErrorContext
from infrastructure.errors.infrastructure_error import InfrastructureError


class HttpClientError(InfrastructureError):
    """Raised for errors related to HTTP client operations."""

    def __init__(
        self,
        message: str,
        url: str | None = None,
        method: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
        request_timeout: float | None = None,
        error_code: str | None = None,
        original_exception: Exception | None = None,
    ):
        context = ErrorContext.empty()
        if url:
            context = context.add_text_entry("url", url)
        if method:
            context = context.add_text_entry("method", method)
        if response_body:
            # Truncate long responses
            context = context.add_text_entry("response_body", response_body[:1000])
        if status_code:
            context = context.add_numeric_entry("status_code", status_code)
        if request_timeout:
            context = context.add_numeric_entry("request_timeout", request_timeout)

        super().__init__(
            message=message,
            error_code=error_code or "HTTP_CLIENT_ERROR",
            context=context,
            original_exception=original_exception,
        )

    def is_retryable(self) -> bool:
        """Determine if HTTP error is retryable based on status code."""
        status_code = self.get_numeric_context_value("status_code")
        if status_code and isinstance(status_code, int):
            # Retry on server errors (5xx) and some client errors
            retryable_codes = {408, 429, 502, 503, 504, 520, 521, 522, 523, 524}
            return status_code >= 500 or status_code in retryable_codes
        return False

    def get_user_friendly_message(self) -> str:
        """Get user-friendly HTTP error message."""
        status_code = self.get_numeric_context_value("status_code")
        url = self.get_text_context_value("url")

        if status_code and url:
            return f"HTTP {status_code} error when accessing {url}"
        elif status_code:
            return f"HTTP {status_code} error"
        elif url:
            return f"HTTP error when accessing {url}"
        else:
            return f"HTTP client error: {self.message}"

    @classmethod
    def timeout_error(
        cls, url: str, timeout: float, method: str = "GET"
    ) -> "HttpClientError":
        """Create error for HTTP timeout."""
        return cls(
            message=f"HTTP request to {url} timed out after {timeout} seconds",
            url=url,
            method=method,
            request_timeout=timeout,
            error_code="HTTP_TIMEOUT",
        )

    @classmethod
    def connection_error(
        cls, url: str, method: str = "GET", original_exception: Exception | None = None
    ) -> "HttpClientError":
        """Create error for HTTP connection failure."""
        return cls(
            message=f"Failed to connect to {url}",
            url=url,
            method=method,
            error_code="HTTP_CONNECTION_ERROR",
            original_exception=original_exception,
        )

    @classmethod
    def status_error(
        cls, url: str, method: str, status_code: int, response_body: str | None = None
    ) -> "HttpClientError":
        """Create error for HTTP status code error."""
        message = f"HTTP {status_code} error for {method} {url}"
        return cls(
            message=message,
            url=url,
            method=method,
            status_code=status_code,
            response_body=response_body,
            error_code="HTTP_STATUS_ERROR",
        )

    @classmethod
    def parse_error(
        cls,
        url: str,
        content_type: str | None = None,
        original_exception: Exception | None = None,
    ) -> "HttpClientError":
        """Create error for HTTP response parsing failure."""
        message = f"Failed to parse HTTP response from {url}"
        if content_type:
            message += f" (content-type: {content_type})"
        return cls(
            message=message,
            url=url,
            error_code="HTTP_PARSE_ERROR",
            original_exception=original_exception,
        )
