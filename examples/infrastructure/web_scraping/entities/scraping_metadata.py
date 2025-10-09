"""Web scraping metadata structure."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ScrapingMetadata:
    """Metadata collected during web scraping operations."""

    page_title: str = ""
    page_description: str = ""
    content_type: str = ""
    content_length: int = 0
    response_status: int = 200
    # response_headers removed - dict type violation
    charset: str = "utf-8"
    language: str = ""
    last_modified: str = ""
    robots_meta: str = ""

    # __post_init__ removed - no dict fields to initialize
