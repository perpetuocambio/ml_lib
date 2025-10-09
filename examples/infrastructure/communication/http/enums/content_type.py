"""HTTP content types enum."""

from enum import Enum


class ContentType(Enum):
    """HTTP content types as enum instead of raw strings."""

    # Text types
    TEXT_PLAIN = "text/plain"
    TEXT_HTML = "text/html"
    TEXT_CSS = "text/css"
    TEXT_JAVASCRIPT = "text/javascript"

    # Application types
    APPLICATION_JSON = "application/json"
    APPLICATION_XML = "application/xml"
    APPLICATION_PDF = "application/pdf"
    APPLICATION_OCTET_STREAM = "application/octet-stream"
    APPLICATION_FORM_URLENCODED = "application/x-www-form-urlencoded"

    # Multipart
    MULTIPART_FORM_DATA = "multipart/form-data"

    # Image types
    IMAGE_JPEG = "image/jpeg"
    IMAGE_PNG = "image/png"
    IMAGE_GIF = "image/gif"
    IMAGE_SVG = "image/svg+xml"
