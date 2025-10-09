"""Advanced web scraping implementation with multi-level support."""

import time
from datetime import datetime
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
from infrastructure.communication.http.client.http_client import HttpClient
from infrastructure.communication.http.headers.http_header import HttpHeader
from infrastructure.communication.http.headers.http_headers_collection import (
    HttpHeadersCollection,
)
from infrastructure.web_scraping.dtos.content_type import ContentType
from infrastructure.web_scraping.dtos.link_info import LinkInfo
from infrastructure.web_scraping.dtos.media_info import MediaInfo
from infrastructure.web_scraping.dtos.scraped_content import ScrapedContent
from infrastructure.web_scraping.dtos.scraping_depth import ScrapingDepth
from infrastructure.web_scraping.dtos.scraping_metadata import (
    ScrapingMetadata,
)
from infrastructure.web_scraping.dtos.user_agent import UserAgent
from infrastructure.web_scraping.entities.scraping_configuration import (
    ScrapingConfiguration,
)

# Infrastructure DTOs - simple data containers without Domain/Application dependencies


class AdvancedWebScraper:
    """Advanced web scraper with multi-level support."""

    def __init__(self):
        """Initialize advanced web scraper."""
        self.http_client = HttpClient()

    def scrape_single_url(
        self, url: str, config: ScrapingConfiguration
    ) -> ScrapedContent:
        """Scrape content from a single URL."""
        start_time = time.time()

        try:
            # Check robots.txt if required
            if config.respect_robots_txt and not self._can_fetch_url(url):
                return self._create_failed_result(url, "Blocked by robots.txt")

            # Set headers
            headers = self._build_headers(config.custom_headers, config.user_agent)

            # Make request
            self.http_client.timeout = config.timeout_seconds
            response = self.http_client.get(url, headers)
            response.raise_for_status()

            # Parse content
            scraped_content = self._parse_response(response, url, config, start_time)

            return scraped_content

        except Exception as e:
            return self._create_failed_result(url, str(e))

    def scrape_with_depth(
        self, url: str, config: ScrapingConfiguration
    ) -> ScrapedContent:
        """Scrape content with specified depth level."""
        visited_urls: set[str] = set()
        pages_to_scrape = [(url, 0)]  # (url, depth_level)
        scraped_pages: list[ScrapedContent] = []

        while pages_to_scrape and len(scraped_pages) < config.max_pages:
            current_url, depth = pages_to_scrape.pop(0)

            if current_url in visited_urls:
                continue

            visited_urls.add(current_url)

            # Scrape current page
            page_result = self.scrape_single_url(current_url, config)

            if page_result.success:
                scraped_pages.append(page_result)

                # Add child links if within depth limit
                if self._should_scrape_deeper(depth, config.depth):
                    child_links = self._filter_links_for_depth_scraping(
                        page_result.links,
                        current_url,
                        config.follow_external_links,
                        visited_urls,
                    )

                    for link in child_links:
                        pages_to_scrape.append((link.url, depth + 1))

            # Respect delay between requests
            if pages_to_scrape and config.delay_between_requests > 0:
                time.sleep(config.delay_between_requests)

        # Create hierarchical result
        if not scraped_pages:
            return self._create_failed_result(url, "No pages could be scraped")

        main_page = scraped_pages[0]
        child_pages = scraped_pages[1:] if len(scraped_pages) > 1 else None

        # Update main page with child information
        main_page.child_pages = child_pages
        main_page.total_pages_scraped = len(scraped_pages)
        main_page.scraping_depth_achieved = self._determine_achieved_depth(
            len(scraped_pages), config.depth
        )

        return main_page

    def validate_url(self, url: str) -> bool:
        """Validate if URL is scrapeable."""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    def get_robots_txt(self, domain: str) -> str | None:
        """Get robots.txt content for domain."""
        try:
            robots_url = f"https://{domain}/robots.txt"
            headers = HttpHeadersCollection()
            headers.add_header(HttpHeader.user_agent("PyIntelCivil-Bot/1.0"))

            self.http_client.timeout = 10
            response = self.http_client.get(robots_url, headers)
            if response.status_code == 200:
                return response.content.decode("utf-8", errors="ignore")
        except Exception:
            pass
        return None

    def estimate_scraping_time(self, url: str, config: ScrapingConfiguration) -> float:
        """Estimate time required for scraping operation."""
        base_time = 5.0  # Base time per page

        if config.depth == ScrapingDepth.SINGLE_PAGE:
            return base_time
        elif config.depth == ScrapingDepth.ONE_LEVEL:
            return base_time * min(5, config.max_pages)  # Estimate 5 child pages
        elif config.depth == ScrapingDepth.TWO_LEVELS:
            return base_time * min(15, config.max_pages)  # Estimate 15 total pages
        else:  # FULL_SITE
            return base_time * config.max_pages

    def _parse_response(
        self,
        response,
        original_url: str,
        config: ScrapingConfiguration,
        start_time: float,
    ) -> ScrapedContent:
        """Parse HTTP response into ScrapedContent."""
        try:
            # Parse HTML
            soup = BeautifulSoup(response.content, "html.parser")

            # Extract title
            title = soup.title.string.strip() if soup.title else "No Title"

            # Extract text content
            content = self._extract_text_content(soup, config)

            # Extract links
            links = self._extract_links(soup, response.url, config)

            # Extract media
            media = self._extract_media(soup, response.url, config)

            # Create metadata
            metadata = ScrapingMetadata(
                scraped_at=datetime.now().isoformat(),
                user_agent="PyIntelCivil-WebScraper/1.0",
                response_time_ms=int((time.time() - start_time) * 1000),
                content_encoding=response.headers.get("content-encoding", "identity"),
                language_detected=self._detect_language(soup),
                page_load_time=time.time() - start_time,
                total_requests=1,
                failed_requests=0,
                robots_txt_respected=config.respect_robots_txt,
            )

            return ScrapedContent(
                success=True,
                url=original_url,
                final_url=response.url,
                title=title,
                content=content,
                content_length=len(content),
                links=links,
                media=media,
                metadata=metadata,
                child_pages=None,  # Set later for multi-level scraping
                total_pages_scraped=1,
                scraping_depth_achieved=ScrapingDepth.SINGLE_PAGE,
            )

        except Exception as e:
            return self._create_failed_result(original_url, f"Parsing failed: {str(e)}")

    def _extract_text_content(self, soup, config: ScrapingConfiguration) -> str:
        """Extract text content based on configuration."""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        # Get text content
        text_content = soup.get_text()

        # Clean up text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text_content = "\n".join(chunk for chunk in chunks if chunk)

        # Truncate if needed
        if len(text_content) > config.max_content_length:
            text_content = text_content[: config.max_content_length] + "... [TRUNCATED]"

        return text_content

    def _extract_links(
        self, soup, base_url: str, config: ScrapingConfiguration
    ) -> list[LinkInfo]:
        """Extract links from HTML."""
        links = []

        if config.content_type in [
            ContentType.TEXT_WITH_LINKS,
            ContentType.FULL_CONTENT,
        ]:
            for link_tag in soup.find_all("a", href=True):
                href = link_tag["href"]
                text = link_tag.get_text(strip=True)

                if not text or not href:
                    continue

                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, href)

                # Determine link type
                link_type = self._classify_link(absolute_url, base_url)

                links.append(
                    LinkInfo(
                        url=absolute_url,
                        text=text,
                        link_type=link_type,
                        depth_level=0,
                        is_scraped=False,
                    )
                )

                # Limit number of links
                if len(links) >= 50:
                    break

        return links

    def _extract_media(
        self, soup, base_url: str, config: ScrapingConfiguration
    ) -> list[MediaInfo]:
        """Extract media information from HTML."""
        media = []

        if config.content_type in [
            ContentType.TEXT_WITH_MEDIA,
            ContentType.FULL_CONTENT,
        ]:
            # Extract images
            for img in soup.find_all("img", src=True):
                src = urljoin(base_url, img["src"])
                alt_text = img.get("alt", "")

                media.append(
                    MediaInfo(
                        url=src,
                        alt_text=alt_text,
                        media_type="image",
                    )
                )

                if len(media) >= 20:  # Limit media items
                    break

        return media

    def _build_headers(
        self, custom_headers: HttpHeadersCollection | None, user_agent: UserAgent
    ) -> HttpHeadersCollection:
        """Build HTTP headers for requests."""
        # Start with browser headers
        headers = HttpHeadersCollection.create_web_browser_headers(user_agent.value)

        # Add custom headers if provided
        if custom_headers:
            for header in custom_headers.get_headers_for_request():
                headers.add_header(header)

        return headers

    def _can_fetch_url(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt."""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc

            robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()

            return rp.can_fetch("*", url)
        except Exception:
            return True  # If can't check, assume it's OK

    def _classify_link(self, link_url: str, base_url: str) -> str:
        """Classify link type (internal, external, etc.)."""
        base_domain = urlparse(base_url).netloc
        link_domain = urlparse(link_url).netloc

        if link_url.startswith("mailto:"):
            return "mailto"
        elif link_url.startswith("tel:"):
            return "tel"
        elif link_domain == base_domain:
            return "internal"
        else:
            return "external"

    def _detect_language(self, soup) -> str | None:
        """Detect page language."""
        # Check html lang attribute
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            return html_tag["lang"]

        # Check meta tags
        lang_meta = soup.find("meta", attrs={"name": "language"})
        if lang_meta and lang_meta.get("content"):
            return lang_meta["content"]

        return None

    def _should_scrape_deeper(
        self, current_depth: int, max_depth: ScrapingDepth
    ) -> bool:
        """Check if should scrape deeper based on configuration."""
        if max_depth == ScrapingDepth.SINGLE_PAGE:
            return False
        elif max_depth == ScrapingDepth.ONE_LEVEL:
            return current_depth < 1
        elif max_depth == ScrapingDepth.TWO_LEVELS:
            return current_depth < 2
        else:  # FULL_SITE
            return current_depth < 10  # Reasonable limit

    def _filter_links_for_depth_scraping(
        self,
        links: list[LinkInfo],
        base_url: str,
        follow_external: bool,
        visited_urls: set[str],
    ) -> list[LinkInfo]:
        """Filter links for depth scraping."""
        filtered_links = []

        for link in links:
            # Skip already visited URLs
            if link.url in visited_urls:
                continue

            # Skip external links if not following them
            if not follow_external and link.link_type == "external":
                continue

            # Skip non-HTTP links
            if not link.url.startswith(("http://", "https://")):
                continue

            filtered_links.append(link)

        return filtered_links[:10]  # Limit to 10 links per page

    def _determine_achieved_depth(
        self, pages_scraped: int, target_depth: ScrapingDepth
    ) -> ScrapingDepth:
        """Determine the actual depth achieved."""
        if pages_scraped == 1:
            return ScrapingDepth.SINGLE_PAGE
        elif pages_scraped <= 5:
            return ScrapingDepth.ONE_LEVEL
        elif pages_scraped <= 20:
            return ScrapingDepth.TWO_LEVELS
        else:
            return ScrapingDepth.FULL_SITE

    def _create_failed_result(self, url: str, error: str) -> ScrapedContent:
        """Create a failed scraping result."""
        return ScrapedContent(
            success=False,
            url=url,
            final_url=url,
            title="",
            content="",
            content_length=0,
            links=[],
            media=[],
            metadata=ScrapingMetadata(
                scraped_at=datetime.now().isoformat(),
                user_agent="AdvancedWebScraper",
                response_time_ms=0,
                content_encoding="",
                total_requests=0,
                failed_requests=1,
                robots_txt_respected=True,
            ),
            error_message=error,
        )
