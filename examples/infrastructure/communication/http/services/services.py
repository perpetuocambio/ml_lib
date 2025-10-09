# Clean HTTP services without dict violations
from infrastructure.communication.http.entities.http_headers import HttpHeaders
from infrastructure.communication.http.entities.http_request_body import HttpRequestBody
from infrastructure.communication.http.entities.requests.header_entry import HeaderEntry
from infrastructure.communication.http.entities.requests.requests_headers import (
    RequestsHeaders,
)
from infrastructure.communication.http.interfaces.http_client_interface import (
    HttpClientInterface,
)
from infrastructure.communication.http.interfaces.http_response import HttpResponse


class RequestsHttpClient(HttpClientInterface):
    """HTTP client using requests library - NO dict exposure."""

    def get(self, url: str, headers: HttpHeaders | None = None) -> HttpResponse:
        """GET request with encapsulated dict usage."""
        requests_headers = RequestsHeaders.from_http_headers(
            headers
        ) or RequestsHeaders(headers_map=[])
        resp = requests_headers.make_get_request(url)
        return HttpResponse(
            resp.status_code, resp.text, HttpHeaders.from_requests_headers(resp.headers)
        )

    def post(
        self,
        url: str,
        body: HttpRequestBody | None = None,
        headers: HttpHeaders | None = None,
    ) -> HttpResponse:
        """POST request with encapsulated dict usage."""
        requests_headers = RequestsHeaders.from_http_headers(
            headers
        ) or RequestsHeaders(headers_map=[])

        # Handle typed request body
        data = None
        if body is not None:
            data = (
                body.get_content_as_string()
                if isinstance(body.content, str)
                else body.get_content_as_bytes()
            )
            # Add content type if needed
            if body.content_type:
                requests_headers = self._add_content_type(
                    requests_headers, body.content_type
                )

        resp = requests_headers.make_post_request(url, data)
        return HttpResponse(
            resp.status_code, resp.text, HttpHeaders.from_requests_headers(resp.headers)
        )

    def put(
        self,
        url: str,
        body: HttpRequestBody | None = None,
        headers: HttpHeaders | None = None,
    ) -> HttpResponse:
        """PUT request with encapsulated dict usage."""
        requests_headers = RequestsHeaders.from_http_headers(
            headers
        ) or RequestsHeaders(headers_map=[])

        # Handle typed request body
        data = None
        if body is not None:
            data = (
                body.get_content_as_string()
                if isinstance(body.content, str)
                else body.get_content_as_bytes()
            )
            # Add content type if needed
            if body.content_type:
                requests_headers = self._add_content_type(
                    requests_headers, body.content_type
                )

        resp = requests_headers.make_put_request(url, data)
        return HttpResponse(
            resp.status_code, resp.text, HttpHeaders.from_requests_headers(resp.headers)
        )

    def delete(self, url: str, headers: HttpHeaders | None = None) -> HttpResponse:
        """DELETE request with encapsulated dict usage."""
        requests_headers = RequestsHeaders.from_http_headers(
            headers
        ) or RequestsHeaders(headers_map=[])
        resp = requests_headers.make_delete_request(url)
        return HttpResponse(
            resp.status_code, resp.text, HttpHeaders.from_requests_headers(resp.headers)
        )

    def _add_content_type(
        self, requests_headers: RequestsHeaders, content_type: str
    ) -> RequestsHeaders:
        """Add content type header if not present."""
        # Check if Content-Type already exists
        for entry in requests_headers.headers_map:
            if entry.name == "Content-Type":
                return requests_headers

        # Add Content-Type
        new_headers = requests_headers.headers_map + [
            HeaderEntry(name="Content-Type", value=content_type)
        ]
        return RequestsHeaders(headers_map=new_headers)
