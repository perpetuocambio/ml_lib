"""Scraping operation result class."""

from dataclasses import dataclass

from infrastructure.web_scraping.dtos.scraped_content import ScrapedContent


@dataclass
class ScrapingResult:
    """Result of a web scraping operation for intelligence collection."""

    success: bool
    content: ScrapedContent | None
    error_message: str | None
    files_saved: list[str] | None

    # Summary information (no dictionaries - use specific fields)
    total_pages_scraped: int = 1
    depth_level_achieved: str = "single_page"
    total_content_length: int = 0
    total_links_found: int = 0
    total_media_found: int = 0
    scraping_time_seconds: float = 0.0
