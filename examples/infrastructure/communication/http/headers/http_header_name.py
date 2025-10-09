"""HTTP header names as enum."""

from enum import Enum


class HttpHeaderName(Enum):
    """Standard HTTP header names."""

    USER_AGENT = "User-Agent"
    ACCEPT = "Accept"
    ACCEPT_LANGUAGE = "Accept-Language"
    ACCEPT_ENCODING = "Accept-Encoding"
    CONNECTION = "Connection"
    UPGRADE_INSECURE_REQUESTS = "Upgrade-Insecure-Requests"
    CACHE_CONTROL = "Cache-Control"
    CONTENT_TYPE = "Content-Type"
    AUTHORIZATION = "Authorization"
    REFERER = "Referer"
    COOKIE = "Cookie"
    HOST = "Host"
