"""Accept header values as enum."""

from enum import Enum


class AcceptValue(Enum):
    """Accept header values."""

    HTML = "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
    JSON = "application/json"
    XML = "application/xml"
    TEXT = "text/plain"
    ALL = "*/*"
