"""Web scraping entities module."""

from infrastructure.web_scraping.entities.infra_scraping_result import (
    InfraScrapingResult,
)

# from infrastructure.config.algorithms.scraping_config import ScrapingConfig  # Import directly to avoid circular imports
from infrastructure.web_scraping.entities.scraping_metadata import (
    ScrapingMetadata,
)
from infrastructure.web_scraping.entities.web_scraper_result import (
    WebScraperResult,
)

__all__ = ["InfraScrapingResult", "ScrapingMetadata", "WebScraperResult"]
