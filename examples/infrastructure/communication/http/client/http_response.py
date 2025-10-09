"""HTTP response wrapper for client operations."""

from infrastructure.communication.http.headers.http_headers_collection import (
    HttpHeadersCollection,
)
from infrastructure.errors.http_client_error import HttpClientError

# Local error handling without external dependencies


class HttpResponse:
    """HTTP response wrapper."""

    def __init__(
        self, url: str, status_code: int, content: bytes, headers: HttpHeadersCollection
    ):
        self.url = url
        self.status_code = status_code
        self.content = content
        self.headers = headers

    def raise_for_status(self) -> None:
        """Raise exception for HTTP errors."""
        if self.status_code >= 400:
            raise HttpClientError(f"HTTP {self.status_code} error for URL: {self.url}")
