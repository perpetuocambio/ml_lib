"""Custom HTTP header for web scraping."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CustomHeader:
    """Custom HTTP header for scraping."""

    name: str
    value: str
