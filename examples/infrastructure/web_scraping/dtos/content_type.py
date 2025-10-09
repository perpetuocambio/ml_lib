"""Content type enumeration."""

from enum import Enum


class ContentType(Enum):
    """Types of content to extract."""

    TEXT_ONLY = "text_only"
    TEXT_WITH_LINKS = "text_with_links"
    TEXT_WITH_MEDIA = "text_with_media"
    FULL_CONTENT = "full_content"  # Text + links + images + metadata
