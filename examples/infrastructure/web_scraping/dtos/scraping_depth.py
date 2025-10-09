"""Scraping depth enumeration."""

from enum import Enum


class ScrapingDepth(Enum):
    """Scraping depth levels."""

    SINGLE_PAGE = "single_page"  # Only the target page
    ONE_LEVEL = "one_level"  # Target page + direct links
    TWO_LEVELS = "two_levels"  # Target + direct + second level links
    FULL_SITE = "full_site"  # Complete site crawling
