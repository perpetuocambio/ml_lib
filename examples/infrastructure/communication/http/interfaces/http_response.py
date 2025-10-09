"""HTTP response class for handling HTTP client responses."""

from __future__ import annotations

from infrastructure.communication.http.entities.http_headers import HttpHeaders


class HttpResponse:
    """Class representing an HTTP response with status, content and headers."""

    def __init__(self, status_code: int, content: str, headers: HttpHeaders):
        """Initialize an HTTP response.

        Args:
            status_code: HTTP status code
            content: Response body content
            headers: HTTP response headers
        """
        self.status_code = status_code
        self.content = content
        self.headers = headers
