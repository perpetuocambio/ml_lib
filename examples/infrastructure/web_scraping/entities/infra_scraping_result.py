"""Infrastructure DTO for web scraping results."""

from dataclasses import dataclass

from infrastructure.web_scraping.entities.scraping_metadata import (
    ScrapingMetadata,
)


@dataclass
class InfraScrapingResult:
    """Infrastructure DTO for web scraping results."""

    url: str
    content: str
    links: list[str]
    media_urls: list[str]
    metadata: ScrapingMetadata
    scraping_time_seconds: float
    success: bool
    error_message: str | None = None
