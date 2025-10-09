"""Web scraping services module."""

from infrastructure.web_scraping.services.advanced_web_scraper import (
    AdvancedWebScraper,
)
from infrastructure.web_scraping.services.web_scraper_service import (
    WebScraperService,
)

__all__ = ["AdvancedWebScraper", "WebScraperService"]
