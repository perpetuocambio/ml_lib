"""Connection header values as enum."""

from enum import Enum


class ConnectionValue(Enum):
    """Connection header values."""

    KEEP_ALIVE = "keep-alive"
    CLOSE = "close"
