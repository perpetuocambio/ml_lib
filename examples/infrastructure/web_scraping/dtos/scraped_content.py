"""Scraped content data class."""

from dataclasses import dataclass

from infrastructure.web_scraping.dtos.link_info import LinkInfo
from infrastructure.web_scraping.dtos.media_info import MediaInfo
from infrastructure.web_scraping.dtos.scraping_depth import ScrapingDepth
from infrastructure.web_scraping.dtos.scraping_metadata import (
    ScrapingMetadata,
)


@dataclass
class ScrapedContent:
    """Complete scraped content result."""

    success: bool
    url: str
    final_url: str
    title: str
    content: str
    content_length: int
    links: list[LinkInfo]
    media: list[MediaInfo]
    metadata: ScrapingMetadata
    error_message: str | None = None

    # Multi-level scraping results
    child_pages: list["ScrapedContent"] | None = None
    total_pages_scraped: int = 1
    scraping_depth_achieved: ScrapingDepth = ScrapingDepth.SINGLE_PAGE
