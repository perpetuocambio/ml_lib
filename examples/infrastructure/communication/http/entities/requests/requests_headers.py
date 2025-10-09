"""Type-safe wrapper for requests library headers."""

from dataclasses import dataclass

import requests
from infrastructure.communication.http.entities.http_headers import HttpHeaders
from infrastructure.communication.http.entities.requests.header_entry import HeaderEntry


@dataclass(frozen=True)
class RequestsHeaders:
    """Type-safe wrapper for requests library headers."""

    headers_map: list[HeaderEntry]

    def _internal_dict_for_requests_only(self):
        """PRIVATE: Internal dict conversion for requests library only."""
        return {entry.name: entry.value for entry in self.headers_map}

    @classmethod
    def from_http_headers(cls, headers: HttpHeaders | None) -> "RequestsHeaders | None":
        """Create from typed HttpHeaders."""
        if headers is None:
            return None
        return cls(
            headers_map=[
                HeaderEntry(name=entry.name, value=entry.value)
                for entry in headers.entries
            ]
        )

    def make_get_request(self, url: str) -> "requests.Response":
        """Encapsulated GET request with headers."""
        return requests.get(
            url, headers=self._internal_dict_for_requests_only(), timeout=10
        )

    def make_post_request(
        self, url: str, data: str | bytes | None = None
    ) -> "requests.Response":
        """Encapsulated POST request with headers."""
        return requests.post(
            url, data=data, headers=self._internal_dict_for_requests_only(), timeout=10
        )

    def make_put_request(
        self, url: str, data: str | bytes | None = None
    ) -> "requests.Response":
        """Encapsulated PUT request with headers."""
        return requests.put(
            url, data=data, headers=self._internal_dict_for_requests_only(), timeout=10
        )

    def make_delete_request(self, url: str) -> "requests.Response":
        """Encapsulated DELETE request with headers."""
        return requests.delete(
            url, headers=self._internal_dict_for_requests_only(), timeout=10
        )
