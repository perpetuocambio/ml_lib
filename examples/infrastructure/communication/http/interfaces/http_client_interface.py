"""Interface for HTTP clients."""

from abc import ABC, abstractmethod

from infrastructure.communication.http.entities.http_headers import HttpHeaders
from infrastructure.communication.http.entities.http_request_body import HttpRequestBody
from infrastructure.communication.http.interfaces.http_response import HttpResponse


class HttpClientInterface(ABC):
    """Abstract interface for HTTP clients."""

    @abstractmethod
    def get(self, url: str, headers: HttpHeaders | None = None) -> HttpResponse:
        """Make a GET request.

        Args:
            url: Request URL
            headers: Request headers

        Returns:
            HTTP response
        """
        ...

    @abstractmethod
    def post(
        self,
        url: str,
        body: HttpRequestBody | None = None,
        headers: HttpHeaders | None = None,
    ) -> HttpResponse:
        """Make a POST request.

        Args:
            url: Request URL
            body: Request body
            headers: Request headers

        Returns:
            HTTP response
        """
        ...

    @abstractmethod
    def put(
        self,
        url: str,
        body: HttpRequestBody | None = None,
        headers: HttpHeaders | None = None,
    ) -> HttpResponse:
        """Make a PUT request.

        Args:
            url: Request URL
            body: Request body
            headers: Request headers

        Returns:
            HTTP response
        """
        ...

    @abstractmethod
    def delete(self, url: str, headers: HttpHeaders | None = None) -> HttpResponse:
        """Make a DELETE request.

        Args:
            url: Request URL
            headers: Request headers

        Returns:
            HTTP response
        """
        ...
