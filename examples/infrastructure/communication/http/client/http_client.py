"""HTTP client for web scraping without external dependencies."""

import socket
import urllib.error
import urllib.request

from infrastructure.communication.http.client.http_response import HttpResponse
from infrastructure.communication.http.headers.http_header import HttpHeader
from infrastructure.communication.http.headers.http_header_name import HttpHeaderName
from infrastructure.communication.http.headers.http_headers_collection import (
    HttpHeadersCollection,
)
from infrastructure.errors.http_client_error import HttpClientError
from infrastructure.serialization.protocol_serializer import ProtocolSerializer


class HttpClient:
    """Simple HTTP client for web scraping."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def get(self, url: str, headers: HttpHeadersCollection) -> HttpResponse:
        """Make GET request."""
        try:
            # Create request with headers
            req = urllib.request.Request(url)

            # Add headers
            for header in headers.get_headers_for_request():
                req.add_header(header.name.value, header.value)

            # Set timeout
            socket.setdefaulttimeout(self.timeout)

            # Make request
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content = response.read()
                status_code = response.getcode()
                response_headers_dict = ProtocolSerializer.serialize_http_headers(
                    response.info()
                )

                # Convert dict to HttpHeadersCollection
                response_headers = HttpHeadersCollection()
                for name, value in response_headers_dict.items():
                    if hasattr(HttpHeaderName, name.upper().replace("-", "_")):
                        header_name = getattr(
                            HttpHeaderName, name.upper().replace("-", "_")
                        )
                    else:
                        # For unknown headers, we'll use a generic approach
                        continue  # Skip unknown headers for now
                    response_headers.add_header(
                        HttpHeader(name=header_name, value=value)
                    )

                return HttpResponse(
                    url=url,
                    status_code=status_code,
                    content=content,
                    headers=response_headers,
                )

        except urllib.error.HTTPError as e:
            empty_headers = HttpHeadersCollection()
            return HttpResponse(
                url=url, status_code=e.code, content=b"", headers=empty_headers
            )
        except Exception as e:
            raise HttpClientError(f"HTTP request failed: {str(e)}") from e
