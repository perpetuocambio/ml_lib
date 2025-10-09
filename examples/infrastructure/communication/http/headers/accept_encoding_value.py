"""Accept-Encoding header values as enum."""

from enum import Enum


class AcceptEncodingValue(Enum):
    """Accept-Encoding header values."""

    GZIP_DEFLATE = "gzip, deflate"
    GZIP_DEFLATE_BR = "gzip, deflate, br"
    IDENTITY = "identity"
