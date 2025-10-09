"""Protocol for objects that behave like HTTP headers."""

from typing import Protocol

from infrastructure.communication.http.entities.header_pair import HeaderPair


class HeadersLikeObject(Protocol):
    """Protocol for objects that behave like headers (have .items() method)."""

    def items(self) -> list[HeaderPair]:
        """Return headers as typed header pairs."""
        ...
