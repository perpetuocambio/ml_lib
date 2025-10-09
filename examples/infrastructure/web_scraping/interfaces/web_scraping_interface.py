"""Web scraping service interface."""

from abc import ABC, abstractmethod

from infrastructure.web_scraping.dtos.scraped_content import ScrapedContent
from infrastructure.web_scraping.entities.scraping_configuration import (
    ScrapingConfiguration,
)


class IWebScrapingService(ABC):
    """Interface for web scraping services."""

    @abstractmethod
    def scrape_single_url(
        self, url: str, config: ScrapingConfiguration
    ) -> ScrapedContent:
        """
        Scrape content from a single URL.

        Args:
            url: URL to scrape
            config: Scraping configuration

        Returns:
            ScrapedContent with results
        """
        pass

    @abstractmethod
    def scrape_with_depth(
        self, url: str, config: ScrapingConfiguration
    ) -> ScrapedContent:
        """
        Scrape content with specified depth level.

        Args:
            url: Starting URL
            config: Scraping configuration including depth

        Returns:
            ScrapedContent with hierarchical results
        """
        pass

    @abstractmethod
    def validate_url(self, url: str) -> bool:
        """
        Validate if URL is scrapeable.

        Args:
            url: URL to validate

        Returns:
            True if URL can be scraped
        """
        pass

    @abstractmethod
    def get_robots_txt(self, domain: str) -> str | None:
        """Get robots.txt content for domain.

        Args:
            domain: Domain to check

        Returns:
            robots.txt content or None
        """
        pass

    @abstractmethod
    def estimate_scraping_time(self, url: str, config: ScrapingConfiguration) -> float:
        """
        Estimate time required for scraping operation.

        Args:
            url: Target URL
            config: Scraping configuration

        Returns:
            Estimated time in seconds
        """
        pass
