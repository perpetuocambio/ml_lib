"""HTTP header representation with typed names and values."""

from dataclasses import dataclass

from infrastructure.communication.http.headers.http_header_name import HttpHeaderName
from infrastructure.communication.http.headers.http_header_pair import HttpHeaderPair
from infrastructure.communication.http.headers.http_header_value import (
    AcceptEncodingValue,
    AcceptLanguageValue,
    AcceptValue,
    ConnectionValue,
)


@dataclass
class HttpHeader:
    """Typed HTTP header with enum-based name and value."""

    name: HttpHeaderName
    value: str

    @classmethod
    def user_agent(cls, user_agent: str) -> "HttpHeader":
        """Create User-Agent header."""
        return cls(name=HttpHeaderName.USER_AGENT, value=user_agent)

    @classmethod
    def accept_html(cls) -> "HttpHeader":
        """Create Accept header for HTML content."""
        return cls(name=HttpHeaderName.ACCEPT, value=AcceptValue.HTML.value)

    @classmethod
    def accept_language_english(cls) -> "HttpHeader":
        """Create Accept-Language header for English."""
        return cls(
            name=HttpHeaderName.ACCEPT_LANGUAGE,
            value=AcceptLanguageValue.ENGLISH_US.value,
        )

    @classmethod
    def accept_encoding_gzip(cls) -> "HttpHeader":
        """Create Accept-Encoding header for gzip/deflate."""
        return cls(
            name=HttpHeaderName.ACCEPT_ENCODING,
            value=AcceptEncodingValue.GZIP_DEFLATE.value,
        )

    @classmethod
    def connection_keep_alive(cls) -> "HttpHeader":
        """Create Connection header for keep-alive."""
        return cls(
            name=HttpHeaderName.CONNECTION, value=ConnectionValue.KEEP_ALIVE.value
        )

    @classmethod
    def upgrade_insecure_requests(cls) -> "HttpHeader":
        """Create Upgrade-Insecure-Requests header."""
        return cls(name=HttpHeaderName.UPGRADE_INSECURE_REQUESTS, value="1")

    def to_pair(self) -> HttpHeaderPair:
        """Convert to pair for compatibility with HTTP libraries."""
        return HttpHeaderPair(name=self.name.value, value=self.value)
