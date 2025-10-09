"""Collection of HTTP headers with typed management."""

from dataclasses import dataclass, field

from infrastructure.communication.http.headers.http_header import HttpHeader
from infrastructure.communication.http.headers.http_header_name import HttpHeaderName


@dataclass
class HttpHeadersCollection:
    """Collection of HTTP headers with typed operations."""

    headers: list[HttpHeader] = field(default_factory=list)

    def add_header(self, header: HttpHeader) -> None:
        """Add a header to the collection."""
        self.headers.append(header)

    def get_header_value(self, name: HttpHeaderName) -> str | None:
        """Get header value by name."""
        for header in self.headers:
            if header.name == name:
                return header.value
        return None

    def has_header(self, name: HttpHeaderName) -> bool:
        """Check if header exists."""
        return self.get_header_value(name) is not None

    def remove_header(self, name: HttpHeaderName) -> None:
        """Remove header by name."""
        self.headers = [h for h in self.headers if h.name != name]

    def get_headers_for_request(self) -> list[HttpHeader]:
        """Get headers list for HTTP request."""
        return self.headers.copy()

    def get_header_pairs(self) -> list[HttpHeader]:
        """Get headers as list of HttpHeader objects."""
        return self.headers.copy()

    @classmethod
    def create_web_browser_headers(cls, user_agent: str) -> "HttpHeadersCollection":
        """Create standard web browser headers."""
        collection = cls()
        collection.add_header(HttpHeader.user_agent(user_agent))
        collection.add_header(HttpHeader.accept_html())
        collection.add_header(HttpHeader.accept_language_english())
        collection.add_header(HttpHeader.accept_encoding_gzip())
        collection.add_header(HttpHeader.connection_keep_alive())
        collection.add_header(HttpHeader.upgrade_insecure_requests())
        return collection
