from dataclasses import dataclass, field

from infrastructure.web_scraping.dtos.scraped_link import ScrapedLink


@dataclass(frozen=True)
class WebScraperResult:
    """Result object for web scraping operations."""

    success: bool
    title: str = ""
    content: str = ""
    content_length: int = 0
    links: list[ScrapedLink] = field(default_factory=list)
    response_status: int = 0
    content_type: str = ""
    final_url: str = ""
    error: str = ""
