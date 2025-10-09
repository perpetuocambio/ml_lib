"""Scraping metadata data class."""

from dataclasses import dataclass


@dataclass
class ScrapingMetadata:
    """Metadata about the scraping operation."""

    scraped_at: str
    user_agent: str
    response_time_ms: int
    content_encoding: str
    language_detected: str | None = None
    page_load_time: float | None = None
    total_requests: int = 1
    failed_requests: int = 0
    robots_txt_respected: bool = True
