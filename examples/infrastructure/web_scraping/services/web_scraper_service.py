"""Web scraping infrastructure service."""

from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from infrastructure.serialization.protocol_serializer import ProtocolSerializer
from infrastructure.web_scraping.dtos.scraped_content import ScrapedContent
from infrastructure.web_scraping.dtos.scraped_link import ScrapedLink
from infrastructure.web_scraping.entities.scraping_configuration import (
    ScrapingConfiguration,
)
from infrastructure.web_scraping.entities.web_scraper_result import (
    WebScraperResult,
)
from infrastructure.web_scraping.interfaces.web_scraping_interface import (
    IWebScrapingService,
)


class WebScraperService(IWebScrapingService):
    """Infrastructure service for web scraping operations."""

    def __init__(self):
        """Initialize web scraper service."""
        pass

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
        # TODO: This implementation returns a WebScraperResult, but the interface
        # requires a ScrapedContent. This needs to be refactored to return the
        # correct, more detailed DTO.
        try:
            # Basic URL validation
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                # This return type is incorrect and needs to be fixed.
                return WebScraperResult(success=False, error="Invalid URL format")

            # Set headers to mimic browser
            headers = ProtocolSerializer.serialize_http_headers(
                {
                    "User-Agent": config.user_agent.value,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                }
            )

            # Make request with timeout
            response = requests.get(
                url, headers=headers, timeout=config.timeout_seconds
            )
            response.raise_for_status()

            # Parse HTML content
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract title
            title = soup.title.string.strip() if soup.title else "No Title"

            # Extract main text content
            for script in soup(["script", "style"]):
                script.decompose()
            text_content = soup.get_text()
            lines = (line.strip() for line in text_content.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text_content = "\n".join(chunk for chunk in chunks if chunk)

            if len(text_content) > config.max_content_length:
                text_content = (
                    text_content[: config.max_content_length] + "... [TRUNCATED]"
                )

            # Extract links
            links = []
            for link in soup.find_all("a", href=True):
                link_url = link["href"]
                link_text = link.get_text(strip=True)
                if link_url.startswith("http") and link_text:
                    links.append(ScrapedLink(url=link_url, text=link_text))
                    if len(links) >= 20:  # Limit number of links
                        break
            # This return type is incorrect and needs to be fixed.
            return WebScraperResult(
                success=True,
                title=title,
                content=text_content,
                content_length=len(text_content),
                links=links,
                response_status=response.status_code,
                content_type=response.headers.get("content-type", "unknown"),
                final_url=response.url,
            )

        except requests.exceptions.RequestException as e:
            return WebScraperResult(success=False, error=f"Request failed: {str(e)}")
        except Exception as e:
            return WebScraperResult(success=False, error=f"Scraping failed: {str(e)}")

    def scrape_with_depth(
        self, url: str, config: ScrapingConfiguration
    ) -> ScrapedContent:
        """Scrape content with configurable depth for recursive scraping."""
        raise NotImplementedError

    def validate_url(self, url: str) -> bool:
        """Validate if URL is accessible and scrapable."""
        raise NotImplementedError

    def get_robots_txt(self, domain: str) -> str | None:
        """Retrieve and parse robots.txt for the given domain."""
        raise NotImplementedError

    def estimate_scraping_time(self, url: str, config: ScrapingConfiguration) -> float:
        """Estimate time required to scrape URL with given configuration."""
        raise NotImplementedError
